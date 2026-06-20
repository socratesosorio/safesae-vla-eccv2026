"""Activation extraction and intervention hooks for OpenVLA/Llama-style backbones."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch


@dataclass
class HookRecord:
    layer: int
    token_position: str
    tensor: torch.Tensor


class ActivationHook:
    """
    Capture last-token activations from multiple transformer layers during generation.

    The hook target path for OpenVLA is:
    `model.language_model.model.layers[layer_idx]`
    """

    def __init__(self, model, layer_indices: list[int]):
        self.model = model
        self.layer_indices = [int(x) for x in layer_indices]
        self._buffers: dict[int, list[torch.Tensor]] = {idx: [] for idx in self.layer_indices}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def register(self) -> None:
        if self._handles:
            return

        for layer_idx in self.layer_indices:
            buffer = self._buffers[layer_idx]

            # Closure guard: bind `buffer` at definition time.
            def make_hook(buf: list[torch.Tensor]):
                def hook_fn(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    last = hidden[:, -1, :].detach().to("cpu", dtype=torch.float16)
                    buf.append(last)

                return hook_fn

            layer = self.model.language_model.model.layers[layer_idx]
            self._handles.append(layer.register_forward_hook(make_hook(buffer)))

    def get_activations(self) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
        d_hidden = hidden_size_from_model(self.model)
        for layer_idx, buffer in self._buffers.items():
            if not buffer:
                out[layer_idx] = torch.empty((0, d_hidden), dtype=torch.float16)
                continue
            out[layer_idx] = torch.stack(buffer, dim=0).squeeze(1).contiguous()
        return out

    def clear(self) -> None:
        for buffer in self._buffers.values():
            buffer.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class InterventionHook:
    """
    Forward hook for SAE-space feature clamping with residual-correction reconstruction.
    """

    def __init__(
        self,
        model,
        sae,
        layer_idx: int,
        feature_indices: list[int],
        scale: float = 0.0,
        norm_factor: float = 1.0,
        feature_scale_map: Mapping[int, float] | None = None,
        feature_value_map: Mapping[int, float] | None = None,
    ):
        self.model = model
        self.sae = sae
        self.layer_idx = int(layer_idx)
        self.feature_indices = list(feature_indices)
        self.scale = float(scale)
        self.norm_factor = float(max(norm_factor, 1e-8))
        self.feature_scale_map = {int(k): float(v) for k, v in (feature_scale_map or {}).items()}
        self.feature_value_map = {int(k): float(v) for k, v in (feature_value_map or {}).items()}
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def register(self) -> None:
        if self._handle is not None:
            return
        layer = self.model.language_model.model.layers[self.layer_idx]
        self._handle = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        last = hidden[:, -1:, :]  # [B,1,D]
        flat = last.squeeze(1).to(dtype=torch.float32)  # [B,D]
        flat_norm = flat / self.norm_factor

        with torch.no_grad():
            features = self.sae.encode(flat_norm)
            modified = apply_feature_intervention(
                features=features,
                feature_indices=self.feature_indices,
                scale=self.scale,
                feature_scale_map=self.feature_scale_map,
                feature_value_map=self.feature_value_map,
            )

            original_recon = self.sae.decode(features)
            modified_recon = self.sae.decode(modified)
            delta = ((modified_recon - original_recon) * self.norm_factor).to(dtype=hidden.dtype)

        new_hidden = hidden.clone()
        new_hidden[:, -1, :] = new_hidden[:, -1, :] + delta
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # Backward-compatible aliases used by older modules.
    def activate(self) -> None:
        self.register()

    def deactivate(self) -> None:
        self.remove()


def hidden_size_from_model(model) -> int:
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        cfg = model.language_model.config
        if hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
    return 4096


def apply_feature_intervention(
    *,
    features: torch.Tensor,
    feature_indices: Sequence[int] | None = None,
    scale: float = 0.0,
    feature_scale_map: Mapping[int, float] | None = None,
    feature_value_map: Mapping[int, float] | None = None,
) -> torch.Tensor:
    """Apply SAE-space feature interventions.

    Backward-compatible behavior uses one shared `scale` across `feature_indices`.
    Newer causal sweeps can pass `feature_scale_map` to assign per-feature scales.
    `feature_value_map` sets absolute feature values and takes precedence over
    scaling; this is useful for class-mean patching when inactive features make
    multiplicative steering uninformative.
    """

    modified = features.clone()
    set_indices: set[int] = set()
    if feature_value_map:
        for feat_idx, feat_value in feature_value_map.items():
            idx = int(feat_idx)
            modified[:, idx] = float(feat_value)
            set_indices.add(idx)

    scaled_indices: set[int] = set()
    if feature_scale_map:
        for feat_idx, feat_scale in feature_scale_map.items():
            idx = int(feat_idx)
            if idx in set_indices:
                continue
            modified[:, idx] *= float(feat_scale)
            scaled_indices.add(idx)

    if feature_indices:
        fallback_indices = [
            int(idx)
            for idx in feature_indices
            if int(idx) not in scaled_indices and int(idx) not in set_indices
        ]
        if fallback_indices:
            modified[:, fallback_indices] *= float(scale)
    return modified


class ActivationRecorder:
    """Backward-compatible multi-layer recorder wrapper."""

    def __init__(self, layers: list[int], token_position: str = "last") -> None:
        self.layers = layers
        self.token_position = token_position
        self.records: dict[int, list[torch.Tensor]] = defaultdict(list)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _select_token(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.token_position in {"last", "action_only"}:
            return hidden[:, -1, :]
        if self.token_position == "all":
            return hidden
        raise ValueError(f"Unknown token_position: {self.token_position}")

    def _build_hook(self, layer_idx: int) -> Callable:
        def hook_fn(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            selected = self._select_token(hidden).detach().cpu()
            self.records[layer_idx].append(selected)

        return hook_fn

    def attach(self, model) -> None:
        base_layers = model.language_model.model.layers
        for layer in self.layers:
            handle = base_layers[layer].register_forward_hook(self._build_hook(layer))
            self._handles.append(handle)

    def pop_layer_stack(self, layer: int) -> torch.Tensor:
        stacked = torch.stack(self.records[layer], dim=0)
        self.records[layer].clear()
        return stacked

    def clear(self) -> None:
        for layer in self.layers:
            self.records[layer].clear()

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
