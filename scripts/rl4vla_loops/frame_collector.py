"""Frame-recording OpenVLA LIBERO collector for the RL4VLA minimal RL-loop experiments.

Subclasses the SafeSAE `RolloutCollector` (official OpenVLA/LIBERO protocol) and adds:
  - per-step raw 256x256 agentview frames (pre-preprocessing, so training can mirror
    the exact official image path),
  - per-step generated action token ids (exact self-imitation targets),
  - optional observation perturbations (DreamAudit grammar) applied to the raw frame
    before official preprocessing -- used for certificate mining and closure replay,
  - optional LoRA adapter loading (peft) merged into the frozen policy for closed-loop
    evaluation of fine-tuned checkpoints.

All heavy lifting (env construction, init states, prompt template, center crop,
strict-success bookkeeping, activation caching) is inherited unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from src.data.rollout_collector import RolloutCollector
from src.utils.hooks import ActivationHook
from src.utils.runtime import save_json


# DreamAudit observation perturbation grammar (vendored verbatim from
# dreamaudit/perturbations/observation.py so this package has no dreamaudit import
# dependency on the cluster; keep in sync with that file).
def parse_observation_perturbation_spec(spec: str) -> dict[str, Any]:
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"Perturbation spec must be name:type[:params...], got {spec!r}")
    name, kind, params = parts[0], parts[1], parts[2:]
    if kind == "identity":
        return {"name": name, "type": kind, "params": {}}
    if kind == "occlusion":
        frac = float(params[0]) if params else 0.20
        return {"name": name, "type": kind, "params": {"fraction": frac}}
    if kind == "brightness":
        factor = float(params[0]) if params else 0.70
        return {"name": name, "type": kind, "params": {"factor": factor}}
    if kind == "shift":
        dx = float(params[0]) if params else 0.08
        dy = float(params[1]) if len(params) > 1 else 0.0
        return {"name": name, "type": kind, "params": {"dx_fraction": dx, "dy_fraction": dy}}
    if kind == "blur":
        radius = float(params[0]) if params else 1.25
        return {"name": name, "type": kind, "params": {"radius": radius}}
    raise ValueError(f"Unknown perturbation type: {kind}")


def apply_observation_perturbation(image: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    from PIL import Image, ImageFilter

    original = np.asarray(image, dtype=np.uint8)
    kind = str(spec["type"])
    params = dict(spec.get("params", {}))
    perturbed = original.copy()
    if kind == "identity":
        pass
    elif kind == "occlusion":
        frac = max(0.0, min(float(params.get("fraction", 0.20)), 0.95))
        h, w = perturbed.shape[:2]
        side = int(round(min(h, w) * frac))
        y0 = max((h - side) // 2, 0)
        x0 = max((w - side) // 2, 0)
        perturbed[y0 : y0 + side, x0 : x0 + side] = 0
    elif kind == "brightness":
        factor = max(0.0, float(params.get("factor", 0.70)))
        perturbed = np.clip(perturbed.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    elif kind == "shift":
        h, w = perturbed.shape[:2]
        dx = int(round(float(params.get("dx_fraction", 0.08)) * w))
        dy = int(round(float(params.get("dy_fraction", 0.0)) * h))
        shifted = np.zeros_like(perturbed)
        src_x0, src_x1 = max(0, -dx), min(w, w - dx)
        dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
        src_y0, src_y1 = max(0, -dy), min(h, h - dy)
        dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
        if src_x0 < src_x1 and src_y0 < src_y1:
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = perturbed[src_y0:src_y1, src_x0:src_x1]
        perturbed = shifted
    elif kind == "blur":
        radius = max(0.0, float(params.get("radius", 1.25)))
        perturbed = np.asarray(
            Image.fromarray(perturbed).filter(ImageFilter.GaussianBlur(radius=radius)),
            dtype=np.uint8,
        )
    else:
        raise ValueError(f"Unknown perturbation type: {kind}")
    return perturbed


class FrameCollector(RolloutCollector):
    def __init__(
        self,
        config: dict,
        *,
        save_frames: bool = True,
        obs_perturb_spec: dict[str, Any] | None = None,
        adapter_dir: str | None = None,
    ) -> None:
        super().__init__(config)
        self._save_frames = bool(save_frames)
        self._obs_spec = obs_perturb_spec
        self._adapter_dir = adapter_dir
        self._frame_stash: list[np.ndarray] = []
        self._token_stash: list[np.ndarray] = []

    # -- model loading with optional LoRA adapter merge ---------------------------------
    def _load_model(self, checkpoint: str) -> None:
        already_loaded = self.current_checkpoint == checkpoint and self.model is not None
        super()._load_model(checkpoint)
        if self._adapter_dir and not already_loaded:
            from peft import PeftModel

            if self.hook is not None:
                self.hook.remove()
                self.hook = None
            peft_model = PeftModel.from_pretrained(self.model, self._adapter_dir)
            self.model = peft_model.merge_and_unload()
            self.model.eval()
            self._refresh_decode_parameters()
            self.hook = ActivationHook(self.model, layer_indices=self.layers)
            self.hook.register()
            print(f"[FrameCollector] merged LoRA adapter from {self._adapter_dir}")

    # -- per-step hooks ------------------------------------------------------------------
    def _prepare_inputs(self, instruction: str, image: np.ndarray, **kwargs: Any) -> dict:
        raw = np.asarray(image, dtype=np.uint8)
        if self._save_frames:
            self._frame_stash.append(raw.copy())
        if self._obs_spec is not None and str(self._obs_spec.get("type", "identity")) != "identity":
            raw = apply_observation_perturbation(raw, self._obs_spec)
        return super()._prepare_inputs(instruction, raw, **kwargs)

    def _predict_action_custom(self, inputs: dict, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        action, token_ids = super()._predict_action_custom(inputs, **kwargs)
        if self._save_frames:
            self._token_stash.append(np.asarray(token_ids, dtype=np.int64))
        return action, token_ids

    # -- episode entry point ---------------------------------------------------------------
    def collect_episode(self, suite: str, task_idx: int, episode_idx: int) -> dict:
        self._frame_stash = []
        self._token_stash = []
        entry = self._default_schedule_entry(suite=suite, task_idx=task_idx, add_noise=0.0)
        entry["replicate_idx"] = int(episode_idx)
        entry["noise_level"] = 0.0
        rollout = self.collect_single_rollout(task_idx, suite, add_noise=0.0, schedule_entry=entry)
        n_steps = int(rollout["actions"].shape[0])
        if self._save_frames:
            frames = np.stack(self._frame_stash[:n_steps]) if self._frame_stash else np.zeros((0,), dtype=np.uint8)
            tokens = np.stack(self._token_stash[:n_steps]) if self._token_stash else np.zeros((0,), dtype=np.int64)
            rollout["frames"] = frames
            rollout["action_token_ids"] = tokens
        rollout["metadata"]["episode_idx"] = int(episode_idx)
        rollout["metadata"]["obs_perturbation"] = self._obs_spec or {"name": "native", "type": "identity", "params": {}}
        return rollout

    def save_episode(self, rollout: dict, output_dir: Path, rollout_id: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_rollout(rollout, output_dir, rollout_id)
        if self._save_frames and "frames" in rollout and rollout["frames"].size:
            payload = {
                "frames": torch.from_numpy(np.ascontiguousarray(rollout["frames"])),
                "action_token_ids": torch.from_numpy(np.ascontiguousarray(rollout["action_token_ids"])),
            }
            save_file(payload, str(output_dir / f"{rollout_id}.frames.safetensors"))
