"""Feature-level clamping and steering interventions for OpenVLA inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InterventionConfig:
    layer_idx: int
    feature_indices: list[int]
    scale: float = 0.0


class FeatureClamper:
    def __init__(self, model, sae, layer_idx: int, feature_indices: list[int], scale: float = 0.0):
        self.model = model
        self.sae = sae
        self.layer_idx = int(layer_idx)
        self.feature_indices = feature_indices
        self.scale = float(scale)
        self.hook: torch.utils.hooks.RemovableHandle | None = None

    def _intervention_hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        last_token = hidden[:, -1, :]

        with torch.no_grad():
            features = self.sae.encode(last_token)
            original_features = features.clone()
            features[:, self.feature_indices] *= self.scale

            reconstructed = self.sae.decode(features)
            original_recon = self.sae.decode(original_features)
            delta = reconstructed - original_recon
            hidden[:, -1, :] = last_token + delta

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def activate(self):
        layer = self.model.language_model.model.layers[self.layer_idx]
        self.hook = layer.register_forward_hook(self._intervention_hook)

    def deactivate(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
