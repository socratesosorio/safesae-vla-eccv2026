"""LoRA fine-tuning of OpenVLA on self-collected LIBERO rollouts (RL4VLA minimal RL loops).

One script, several arms (sparse-return offline RL = success-filtered self-imitation
is the shared baseline):

  BC (baseline, all three papers):
    --filter success
  nla4vla treatment (+ demonstration-anchored action-MSE auxiliary):
    --filter success --aux-mse-weight 1.0
  safesae treatment (SAE-potential advantage-weighted regression over ALL episodes):
    --filter all --step-weight-mode phi --sae-checkpoint <ckpt>
  dreamaudit treatment (certificate-anchored perturbation augmentation):
    --filter success --cert-json certs_train.json --aug-prob 0.5
  dreamaudit control (size-matched random-perturbation augmentation):
    --filter success --cert-json random --aug-prob 0.5 --random-aug-seed 7

Training target = the policy's own generated action tokens on reward-labeled rollouts
(exact self-imitation; no normalization roundtrip). CE is computed manually from
teacher-forced logits at the action-token positions; the optional auxiliary regresses
the softmax-expected normalized action value toward the demonstrated bin value
(the "demonstration-anchored action-MSE auxiliary" of the return-geometry paper).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.rl4vla_loops.frame_collector import (  # noqa: E402
    apply_observation_perturbation,
    parse_observation_perturbation_spec,
)
from src.data.openvla_action_utils import (  # noqa: E402
    OPENVLA_EMPTY_ACTION_TOKEN_ID,
    apply_center_crop,
    format_openvla_action_prompt,
    preprocess_libero_image,
)

NUM_PATCHES = 256  # prismatic inserts projected patch embeddings after the BOS token
ACTION_DIM = 7

RANDOM_AUG_FAMILIES = [
    ("occlusion", 0.15, 0.55),
    ("brightness", 0.35, 0.80),
    ("shift", 0.04, 0.16),
    ("blur", 0.8, 3.0),
]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
@dataclass
class EpisodeRecord:
    rollout_id: str
    json_path: Path
    frames_path: Path
    tensors_path: Path
    suite: str
    task_idx: int
    instruction: str
    success: bool
    num_steps: int


def scan_episodes(data_dirs: list[Path]) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for d in data_dirs:
        for jp in sorted(d.glob("*.json")):
            if jp.name.startswith("collect_summary") or jp.name.startswith("collection_"):
                continue
            rid = jp.stem
            fp = d / f"{rid}.frames.safetensors"
            tp = d / f"{rid}.safetensors"
            if not fp.exists() or not tp.exists():
                continue
            meta = json.loads(jp.read_text())
            records.append(
                EpisodeRecord(
                    rollout_id=rid,
                    json_path=jp,
                    frames_path=fp,
                    tensors_path=tp,
                    suite=str(meta.get("suite", "object")),
                    task_idx=int(meta.get("task_idx", -1)),
                    instruction=str(meta.get("instruction", "")),
                    success=bool(meta.get("episode_success", False)),
                    num_steps=int(meta.get("num_steps", 0)),
                )
            )
    return records


class StepDataset(torch.utils.data.Dataset):
    """Flat (episode, step) dataset reading frames/token-ids lazily from safetensors."""

    def __init__(
        self,
        episodes: list[EpisodeRecord],
        *,
        image_size: int = 224,
        center_crop: bool = True,
        center_crop_fraction: float = 0.9,
        step_weights: dict[str, np.ndarray] | None = None,
        cert_specs_by_task: dict[tuple[str, int], list[dict]] | None = None,
        aug_prob: float = 0.0,
        aug_jitter: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.episodes = episodes
        self.image_size = image_size
        self.center_crop = center_crop
        self.center_crop_fraction = center_crop_fraction
        self.step_weights = step_weights or {}
        self.cert_specs_by_task = cert_specs_by_task
        self.aug_prob = float(aug_prob)
        self.aug_jitter = float(aug_jitter)
        self.rng = random.Random(seed)
        self.index: list[tuple[int, int]] = []
        for ei, ep in enumerate(episodes):
            for t in range(ep.num_steps):
                self.index.append((ei, t))

    def __len__(self) -> int:
        return len(self.index)

    def _maybe_augment(self, ep: EpisodeRecord, frame: np.ndarray) -> np.ndarray:
        if self.cert_specs_by_task is None or self.rng.random() >= self.aug_prob:
            return frame
        specs = self.cert_specs_by_task.get((ep.suite, ep.task_idx))
        if not specs:
            return frame
        spec = dict(self.rng.choice(specs))
        params = dict(spec.get("params", {}))
        jit = 1.0 + self.rng.uniform(-self.aug_jitter, self.aug_jitter)
        params = {k: float(v) * jit for k, v in params.items()}
        spec["params"] = params
        return apply_observation_perturbation(frame, spec)

    def __getitem__(self, i: int):
        ei, t = self.index[i]
        ep = self.episodes[ei]
        with safe_open(str(ep.frames_path), framework="np") as f:
            frame = f.get_slice("frames")[t]
            tokens = f.get_slice("action_token_ids")[t]
        frame = np.asarray(frame, dtype=np.uint8)
        frame = self._maybe_augment(ep, frame)
        pil = preprocess_libero_image(frame, resize_size=self.image_size)
        if self.center_crop:
            pil = apply_center_crop(pil, crop_fraction=self.center_crop_fraction)
        w = self.step_weights.get(ep.rollout_id)
        weight = float(w[t]) if w is not None and t < len(w) else 1.0
        return {
            "image": pil,
            "instruction": ep.instruction,
            "action_tokens": np.asarray(tokens, dtype=np.int64).reshape(-1)[:ACTION_DIM],
            "weight": weight,
        }


def make_collate(processor):
    tokenizer = processor.tokenizer

    def collate(batch):
        input_ids_list = []
        for ex in batch:
            prompt = format_openvla_action_prompt(ex["instruction"])
            ids = tokenizer(prompt, return_tensors=None, add_special_tokens=True)["input_ids"]
            ids = list(ids) + [OPENVLA_EMPTY_ACTION_TOKEN_ID] + list(int(x) for x in ex["action_tokens"])
            input_ids_list.append(ids)
        max_len = max(len(x) for x in input_ids_list)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id or 0
        input_ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        action_pos = torch.zeros((len(batch), ACTION_DIM), dtype=torch.long)
        for bi, ids in enumerate(input_ids_list):
            input_ids[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[bi, : len(ids)] = 1
            start = len(ids) - ACTION_DIM
            action_pos[bi] = torch.arange(start, len(ids))
        images = [ex["image"] for ex in batch]
        pixel_values = processor.image_processor(images, return_tensors="pt")["pixel_values"]
        weights = torch.tensor([ex["weight"] for ex in batch], dtype=torch.float32)
        tokens = torch.stack([torch.from_numpy(np.asarray(ex["action_tokens"], dtype=np.int64)) for ex in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "pixel_values": pixel_values,
            "action_pos": action_pos,
            "action_tokens": tokens,
            "weights": weights,
        }

    return collate


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------
def action_losses(model, batch, device, dtype, vocab_size: int, value_of_slice: torch.Tensor):
    """Teacher-forced CE + expected-value action MSE at the 7 action-token positions."""
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
    out = model(input_ids=input_ids, attention_mask=attn, pixel_values=pixel_values)
    logits = out.logits  # [B, 256 + L, V]
    action_pos = batch["action_pos"].to(device)  # text-index of each action token
    # text index k (k >= 1) sits at multimodal index 256 + k; its predictive logits
    # are at multimodal index 255 + k.
    gather_idx = action_pos + NUM_PATCHES - 1
    bidx = torch.arange(logits.shape[0], device=device).unsqueeze(1)
    sel = logits[bidx, gather_idx]  # [B, 7, V]
    targets = batch["action_tokens"].to(device)  # [B, 7]
    ce = F.cross_entropy(
        sel.reshape(-1, sel.shape[-1]).float(), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    ce_per_sample = ce.mean(dim=1)
    # expected normalized action value over the action-token slice
    slice_logits = sel[..., vocab_size - 256 : vocab_size].float()
    probs = F.softmax(slice_logits, dim=-1)
    expected = (probs * value_of_slice.to(device)).sum(dim=-1)  # [B, 7]
    target_vals = value_of_slice.to(device)[torch.clamp(targets - (vocab_size - 256), 0, 255)]
    mse_per_sample = ((expected - target_vals) ** 2).mean(dim=1)
    with torch.no_grad():
        acc = (sel.argmax(dim=-1) == targets).float().mean()
    return ce_per_sample, mse_per_sample, acc


# --------------------------------------------------------------------------------------
# Phi (SAE progress potential) weights
# --------------------------------------------------------------------------------------
def compute_phi_weights(
    episodes: list[EpisodeRecord],
    sae_checkpoint: str,
    *,
    layer: int = 20,
    gamma: float = 0.99,
    device: str = "cuda",
    out_json: Path | None = None,
) -> dict[str, np.ndarray]:
    from src.analysis.differential_activation import load_sae_checkpoint

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    sae, _norm = load_sae_checkpoint(sae_checkpoint, d_in=4096, d_sae=16384, k=32, device=dev)
    feats: dict[str, torch.Tensor] = {}
    labels: dict[str, float] = {}
    for ep in episodes:
        with safe_open(str(ep.tensors_path), framework="pt") as f:
            acts = f.get_tensor(f"activations_layer{layer}")  # [T, 7, 4096] fp16
        x = acts.float().mean(dim=1).to(dev)
        with torch.no_grad():
            z = sae.encode(x)
        feats[ep.rollout_id] = z.cpu()
        labels[ep.rollout_id] = 1.0 if ep.success else 0.0

    # torch logistic readout: step features -> episode success
    X = torch.cat([feats[ep.rollout_id] for ep in episodes]).to(dev)
    y = torch.cat(
        [torch.full((feats[ep.rollout_id].shape[0],), labels[ep.rollout_id]) for ep in episodes]
    ).to(dev)
    lin = torch.nn.Linear(X.shape[1], 1).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    n = X.shape[0]
    for epoch in range(8):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, 8192):
            idx = perm[s : s + 8192]
            opt.zero_grad()
            logit = lin(X[idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        phi_all = torch.sigmoid(lin(X).squeeze(-1)).cpu().numpy()
        # episode-level sanity AUROC
        ep_means, ep_labels = [], []
        off = 0
        phis: dict[str, np.ndarray] = {}
        for ep in episodes:
            T = feats[ep.rollout_id].shape[0]
            phis[ep.rollout_id] = phi_all[off : off + T]
            ep_means.append(float(phi_all[off : off + T].mean()))
            ep_labels.append(labels[ep.rollout_id])
            off += T
    order = np.argsort(ep_means)
    ranks = np.empty(len(order)); ranks[order] = np.arange(len(order))
    pos = [r for r, l in zip(ranks, ep_labels) if l > 0.5]
    neg = [r for r, l in zip(ranks, ep_labels) if l <= 0.5]
    auroc = float("nan")
    if pos and neg:
        auroc = (np.mean(pos) - (len(pos) - 1) / 2.0) / len(neg) if len(neg) else float("nan")
        auroc = float((sum(pos) - len(pos) * (len(pos) - 1) / 2.0) / (len(pos) * len(neg)))

    # potential-based advantage -> exp weights
    adv_all: dict[str, np.ndarray] = {}
    flat = []
    for ep in episodes:
        phi = phis[ep.rollout_id]
        T = len(phi)
        adv = np.zeros(T, dtype=np.float32)
        if T > 1:
            adv[:-1] = gamma * phi[1:] - phi[:-1]
        adv[-1] = (1.0 if ep.success else 0.0) - phi[-1]
        adv_all[ep.rollout_id] = adv
        flat.append(adv)
    flat = np.concatenate(flat)
    tau = max(float(np.std(flat)), 1e-4)
    weights: dict[str, np.ndarray] = {}
    all_w = []
    for rid, adv in adv_all.items():
        w = np.clip(np.exp(adv / tau), 0.1, 10.0)
        weights[rid] = w
        all_w.append(w)
    mean_w = float(np.concatenate(all_w).mean())
    weights = {rid: (w / mean_w).astype(np.float32) for rid, w in weights.items()}

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(
                {
                    "tau": tau,
                    "episode_phi_auroc": auroc,
                    "gamma": gamma,
                    "weights": {k: v.tolist() for k, v in weights.items()},
                }
            )
        )
    print(f"[phi] step-feature episode AUROC={auroc:.3f} tau={tau:.4f}")
    return weights


# --------------------------------------------------------------------------------------
# Cert specs
# --------------------------------------------------------------------------------------
def load_cert_specs(cert_json: str, episodes: list[EpisodeRecord], seed: int) -> dict[tuple[str, int], list[dict]]:
    tasks = sorted({(ep.suite, ep.task_idx) for ep in episodes})
    if cert_json == "random":
        rng = random.Random(seed)
        out: dict[tuple[str, int], list[dict]] = {}
        for key in tasks:
            specs = []
            for fam, lo, hi in RANDOM_AUG_FAMILIES:
                for _ in range(2):
                    v = rng.uniform(lo, hi)
                    spec_str = f"rand_{fam}:{fam}:{v:.3f}"
                    specs.append(parse_observation_perturbation_spec(spec_str))
            out[key] = specs
        return out
    data = json.loads(Path(cert_json).read_text())
    out = {}
    for c in data["certificates"]:
        key = (str(c["suite"]), int(c["task_idx"]))
        out.setdefault(key, []).append({"name": c["name"], "type": c["type"], "params": c["params"]})
    return out


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-object")
    ap.add_argument("--code-revision", default="47a0ec7fc4ec123775a391911046cf33cf9ed83f")
    ap.add_argument("--filter", choices=["success", "all"], default="success")
    ap.add_argument("--aux-mse-weight", type=float, default=0.0)
    ap.add_argument("--step-weight-mode", choices=["none", "phi"], default="none")
    ap.add_argument("--sae-checkpoint", default="/work/joy/safesae-vla/checkpoints/task240_2x_medium_layer20/sae_layer20_d16384.pt")
    ap.add_argument("--cert-json", default=None, help="certs_train.json path or the literal 'random'")
    ap.add_argument("--aug-prob", type=float, default=0.5)
    ap.add_argument("--random-aug-seed", type=int, default=7)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="alignment smoke test only")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    episodes = scan_episodes([Path(d) for d in args.data_dirs])
    print(f"[data] {len(episodes)} episodes, {sum(e.success for e in episodes)} successes")
    # episode-level train/val split (deterministic)
    rng = random.Random(1234)
    eps = sorted(episodes, key=lambda e: e.rollout_id)
    rng.shuffle(eps)
    n_val = max(2, int(len(eps) * args.val_fraction))
    val_eps_all = eps[:n_val]
    train_eps_all = eps[n_val:]
    if args.filter == "success":
        train_eps = [e for e in train_eps_all if e.success]
    else:
        train_eps = list(train_eps_all)
    val_eps = [e for e in val_eps_all if e.success]  # validation always on success steps
    print(f"[data] train eps={len(train_eps)} val eps={len(val_eps)}")

    step_weights = None
    if args.step_weight_mode == "phi":
        step_weights = compute_phi_weights(
            train_eps, args.sae_checkpoint, out_json=out_dir / "phi_weights.json"
        )

    cert_specs = None
    if args.cert_json:
        cert_specs = load_cert_specs(args.cert_json, train_eps, args.random_aug_seed)
        n_specs = sum(len(v) for v in cert_specs.values())
        print(f"[aug] cert specs for {len(cert_specs)} tasks, {n_specs} specs total")

    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True, use_fast=False, code_revision=args.code_revision
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        code_revision=args.code_revision,
    ).to(device)
    vocab_size = int(
        getattr(model, "vocab_size", None)
        or getattr(model.config, "vocab_size", None)
        or model.config.text_config.vocab_size
    )
    bin_centers = np.linspace(-1.0, 1.0, 256)
    # token id (vocab-256+b) decodes to bin_centers[clip(vocab - id - 1, 0, 255)] = bin_centers[255-b]
    value_of_slice = torch.tensor(bin_centers[::-1].copy(), dtype=torch.float32)

    train_ds = StepDataset(
        train_eps,
        step_weights=step_weights,
        cert_specs_by_task=cert_specs,
        aug_prob=args.aug_prob if cert_specs else 0.0,
        seed=args.seed,
    )
    val_ds = StepDataset(val_eps)
    collate = make_collate(processor)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate
    )

    if args.smoke:
        model.eval()
        batch = next(iter(val_loader if len(val_ds) else train_loader))
        with torch.no_grad():
            ce, mse, acc = action_losses(model, batch, device, dtype, vocab_size, value_of_slice)
        print(f"[smoke] CE={ce.mean().item():.4f} MSE={mse.mean().item():.5f} tokenacc={acc.item():.3f}")
        # token accuracy should be high (>0.8) if logits/positions are aligned, since
        # targets are the model's own greedy generations.
        return 0 if acc.item() > 0.6 else 1

    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception as exc:  # remote code may not support it
        print(f"[warn] gradient checkpointing unavailable: {exc}")
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        return args.lr

    metrics_path = out_dir / "metrics.jsonl"
    log_f = metrics_path.open("a")

    def run_val(step):
        model.eval()
        ces, mses, accs = [], [], []
        with torch.no_grad():
            for vb in val_loader:
                ce, mse, acc = action_losses(model, vb, device, dtype, vocab_size, value_of_slice)
                ces.append(ce.mean().item()); mses.append(mse.mean().item()); accs.append(acc.item())
                if len(ces) >= 60:
                    break
        model.train()
        row = {
            "step": step, "split": "val",
            "ce": float(np.mean(ces)) if ces else None,
            "action_mse": float(np.mean(mses)) if mses else None,
            "token_acc": float(np.mean(accs)) if accs else None,
            "time": time.time(),
        }
        log_f.write(json.dumps(row) + "\n"); log_f.flush()
        print(f"[val @{step}] ce={row['ce']:.4f} mse={row['action_mse']:.5f} acc={row['token_acc']:.3f}")

    run_val(0)
    step = 0
    t0 = time.time()
    accum_ce, accum_mse, accum_n = 0.0, 0.0, 0
    data_iter = iter(train_loader)
    while step < args.max_steps:
        opt.zero_grad()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            ce, mse, _acc = action_losses(model, batch, device, dtype, vocab_size, value_of_slice)
            w = batch["weights"].to(device)
            loss = (w * ce).mean() + args.aux_mse_weight * (w * mse).mean()
            (loss / args.grad_accum).backward()
            accum_ce += float(ce.mean().item()); accum_mse += float(mse.mean().item()); accum_n += 1
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % 25 == 0:
            row = {
                "step": step, "split": "train",
                "ce": accum_ce / max(accum_n, 1),
                "action_mse": accum_mse / max(accum_n, 1),
                "lr": lr_at(step),
                "samples_per_s": (25 * args.grad_accum * args.batch_size) / max(time.time() - t0, 1e-6),
                "time": time.time(),
            }
            log_f.write(json.dumps(row) + "\n"); log_f.flush()
            print(f"[train @{step}] ce={row['ce']:.4f} mse={row['action_mse']:.5f} {row['samples_per_s']:.2f} samp/s")
            accum_ce = accum_mse = 0.0; accum_n = 0; t0 = time.time()
        if step % args.eval_every == 0:
            run_val(step)
        if step % args.save_every == 0 or step == args.max_steps:
            ck = out_dir / f"adapter_step{step:06d}"
            model.save_pretrained(str(ck))
            print(f"[save] {ck}")

    model.save_pretrained(str(out_dir / "adapter_final"))
    (out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=1, default=str))
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
