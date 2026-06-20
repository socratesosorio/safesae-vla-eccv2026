"""Fast mechanistic causal analysis via SAE-space activation patching."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq

from src.data.activation_dataset import AnalysisDataset
from src.sae.model import BatchTopKSAE
from src.sae.train_sae import BatchTopKSAE as LegacyBatchTopKSAE
from src.utils.config import load_yaml
from src.utils.runtime import ensure_dir


class ActivationPatcher:
    """
    Fast activation-space intervention analysis.

    For cached layer activations x:
      features = SAE.encode(x)
      features' = clamp(features)
      x' = SAE.decode(features')
      delta = x' - SAE.decode(features)

    Mechanistic effect proxy on action logits:
      delta_logits = delta @ lm_head_weight.T
    """

    def __init__(self, model, sae: BatchTopKSAE, layer_idx: int = 20, norm_factor: float = 1.0):
        self.model = model
        self.sae = sae
        self.layer_idx = int(layer_idx)
        self.norm_factor = float(max(norm_factor, 1e-8))
        lm_head = getattr(model.language_model, "lm_head", None) if hasattr(model, "language_model") else None
        if lm_head is None:
            lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            raise ValueError("Model does not expose lm_head weights")
        self.lm_head_weight = lm_head.weight.detach()
        sae_d_in = int(getattr(self.sae, "d_in", self.lm_head_weight.shape[1]))
        if int(self.lm_head_weight.shape[1]) != sae_d_in:
            raise ValueError(
                f"lm_head hidden size ({self.lm_head_weight.shape[1]}) does not match SAE d_in ({sae_d_in})."
            )

    @torch.no_grad()
    def patch_and_predict(
        self,
        activation: torch.Tensor,
        feature_indices: list[int],
        scale: float = 0.0,
    ) -> tuple[int, int, np.ndarray]:
        act_raw = activation.reshape(1, -1).to(dtype=torch.float32, device=next(self.sae.parameters()).device)
        act_norm = act_raw / self.norm_factor
        feats = self.sae.encode(act_norm)
        feats_mod = feats.clone()
        feats_mod[:, feature_indices] *= float(scale)

        recon_norm = self.sae.decode(feats)
        recon_mod_norm = self.sae.decode(feats_mod)
        delta = ((recon_mod_norm - recon_norm) * self.norm_factor).squeeze(0)

        w = self.lm_head_weight.to(delta.device, dtype=delta.dtype)
        baseline_logits = act_raw.squeeze(0) @ w.T
        delta_logits = delta @ w.T
        patched_logits = baseline_logits + delta_logits
        vocab_size = w.shape[0]
        action_start = max(vocab_size - 256, 0)
        baseline_slice = baseline_logits[action_start:]
        patched_slice = patched_logits[action_start:]
        if baseline_slice.numel() == 0 or patched_slice.numel() == 0:
            raise RuntimeError("lm_head vocabulary is too small to derive action-bin slice")
        original_token = int(action_start + torch.argmax(baseline_slice).item())
        patched_token = int(action_start + torch.argmax(patched_slice).item())

        action_bins = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
        orig_idx = int(np.clip(original_token - action_start, 0, 255))
        patch_idx = int(np.clip(patched_token - action_start, 0, 255))
        delta_scalar = float(action_bins[patch_idx] - action_bins[orig_idx])
        action_delta = np.full((7,), delta_scalar, dtype=np.float32)
        return original_token, patched_token, action_delta

    @torch.no_grad()
    def run_patching_analysis(
        self,
        dataset: AnalysisDataset,
        feature_ranking: pd.DataFrame,
        top_k_values: list[int] | None = None,
        scale: float = 0.0,
    ) -> dict[str, pd.DataFrame]:
        top_k_values = top_k_values or [1, 3, 5, 10, 20]
        feature_ids = feature_ranking["feature_idx"].astype(int).tolist()

        rows = []
        for ep_idx in tqdm(dataset.get_unsafe_episodes(split="all"), desc="Activation patching"):
            item = dataset[ep_idx]
            act_key = f"activations_layer{self.layer_idx}"
            if act_key not in item:
                raise KeyError(f"Missing {act_key} in analysis episode payload")
            acts = item[act_key].to(torch.float32)  # [T,N,d_in]
            pooled = acts.mean(dim=1)  # [T,4096]

            for t in range(pooled.shape[0]):
                activation_t = pooled[t]
                for top_k in top_k_values:
                    selected = feature_ids[: int(top_k)]
                    _, _, delta = self.patch_and_predict(
                        activation=activation_t,
                        feature_indices=selected,
                        scale=scale,
                    )
                    rows.append(
                        {
                            "episode_idx": int(ep_idx),
                            "timestep": int(t),
                            "top_k": int(top_k),
                            "dim0_delta": float(delta[0]),
                            "dim1_delta": float(delta[1]),
                            "dim2_delta": float(delta[2]),
                            "dim3_delta": float(delta[3]),
                            "dim4_delta": float(delta[4]),
                            "dim5_delta": float(delta[5]),
                            "dim6_delta": float(delta[6]),
                            "total_delta_magnitude": float(np.linalg.norm(delta)),
                        }
                    )

        delta_df = pd.DataFrame(rows)
        summary_rows = []
        if not delta_df.empty:
            for top_k, grp in delta_df.groupby("top_k"):
                dim_cols = [f"dim{i}_delta" for i in range(7)]
                means = grp[dim_cols].abs().mean(axis=0).to_numpy()
                summary_rows.append(
                    {
                        "top_k": int(top_k),
                        "mean_delta": float(grp["total_delta_magnitude"].mean()),
                        "std_delta": float(grp["total_delta_magnitude"].std(ddof=0)),
                        "frac_large_delta": float((grp["total_delta_magnitude"] > 0.1).mean()),
                        "most_affected_dimension": int(np.argmax(means)),
                    }
                )
        summary_df = pd.DataFrame(summary_rows)
        return {"action_deltas": delta_df, "summary": summary_df}


def _load_sae_checkpoint(path: str, cfg: dict, device: torch.device) -> tuple[BatchTopKSAE, float]:
    ckpt = torch.load(path, map_location=device)
    primary = cfg.get("primary", cfg.get("sae", cfg))
    d_in = int(ckpt.get("d_in", primary.get("d_in", 4096)))
    d_sae = int(ckpt.get("d_sae", primary.get("d_sae", 16384)))
    k = int(ckpt.get("k", primary.get("k", 32)))
    state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    last_exc: Exception | None = None
    model = None
    for cls in (BatchTopKSAE, LegacyBatchTopKSAE):
        try:
            model = cls(d_in=d_in, d_sae=d_sae, k=k).to(device)  # type: ignore[call-arg]
            model.load_state_dict(state)
            model.eval()
            break
        except Exception as exc:
            last_exc = exc
            model = None
    if model is None:
        raise RuntimeError(f"Unable to load SAE checkpoint {path}: {last_exc}")
    norm_factor = float(ckpt.get("norm_factor", 1.0))
    return model, norm_factor


def _resolve_model_checkpoint(rollout_cfg: dict) -> str | None:
    model_cfg = rollout_cfg.get("model", {})
    checkpoints = model_cfg.get("checkpoints")
    if isinstance(checkpoints, dict) and checkpoints:
        for key in ("spatial", "object", "goal", "long"):
            if key in checkpoints and str(checkpoints[key]).strip():
                return str(checkpoints[key])
        first = next(iter(checkpoints.values()), None)
        if first:
            return str(first)
    for key in ("name", "base_model"):
        value = model_cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _load_model_with_lm_head(checkpoint: str, device: torch.device):
    errors: list[str] = []
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    for loader in (AutoModelForVision2Seq, AutoModelForCausalLM):
        try:
            model = loader.from_pretrained(
                checkpoint,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            ).to(device)
            model.eval()
            return model
        except Exception as exc:  # pragma: no cover - backend/model-availability dependent.
            errors.append(f"{loader.__name__}: {exc}")
    raise RuntimeError(f"Unable to load a model with lm_head from {checkpoint}. Errors: {' | '.join(errors)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run activation patching analysis")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--ranked_features", type=str, required=True)
    parser.add_argument("--sae_config", type=str, default="configs/sae.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/eval.yaml")
    parser.add_argument("--rollout_config", type=str, default="configs/rollout.yaml")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="",
        help="Optional HF checkpoint used to source true lm_head logits for patching.",
    )
    parser.add_argument("--output_dir", type=str, default="results/analysis")
    parser.add_argument("--layer", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sae_cfg = load_yaml(args.sae_config)
    eval_cfg = load_yaml(args.eval_config)
    out_dir = ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = _load_sae_checkpoint(args.sae_checkpoint, sae_cfg, device=device)
    dataset = AnalysisDataset(args.data_dir, test_split=float(eval_cfg.get("analysis", {}).get("test_split", 0.2)))
    ranked = pd.read_csv(args.ranked_features)

    rollout_cfg = load_yaml(args.rollout_config) if Path(args.rollout_config).exists() else {}
    checkpoint = str(args.model_checkpoint).strip() or (_resolve_model_checkpoint(rollout_cfg) or "")
    if not checkpoint:
        raise ValueError(
            "Unable to resolve policy checkpoint for activation patching. "
            "Pass --model_checkpoint or provide model checkpoints in rollout config."
        )
    model = _load_model_with_lm_head(checkpoint=checkpoint, device=device)
    layer_idx = int(args.layer if args.layer is not None else eval_cfg.get("activation_patching", {}).get("start_layer", 20))
    patcher = ActivationPatcher(model=model, sae=sae, layer_idx=layer_idx, norm_factor=norm_factor)
    top_k_values = [int(v) for v in eval_cfg.get("activation_patching", {}).get("features_to_test", [1, 3, 5, 10, 20])]
    scale = float(eval_cfg.get("activation_patching", {}).get("scale_factors", [0.0])[0])

    outputs = patcher.run_patching_analysis(dataset=dataset, feature_ranking=ranked, top_k_values=top_k_values, scale=scale)
    outputs["action_deltas"].to_csv(Path(out_dir) / "action_deltas.csv", index=False)
    outputs["summary"].to_csv(Path(out_dir) / "action_deltas_summary.csv", index=False)


if __name__ == "__main__":
    main()
