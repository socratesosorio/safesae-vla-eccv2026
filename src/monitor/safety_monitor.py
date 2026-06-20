"""Runtime safety monitors based on SAE feature representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.metrics import cost_weighted_f1


class RawActivationMLP(nn.Module):
    def __init__(self, d_in: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class MonitorMetrics:
    auroc: float
    f1: float
    precision: float
    recall: float
    cost_weighted_f1: float
    pr_auc: float


class SAEFeatureSafetyMonitor:
    def __init__(self, sae, feature_weights: np.ndarray | None = None, threshold: float = 0.5):
        self.sae = sae
        self.feature_weights = feature_weights
        self.threshold = threshold
        self.lr_model: LogisticRegression | None = None

    @torch.no_grad()
    def extract_features(self, activations: torch.Tensor) -> np.ndarray:
        """activations: [7, 4096] or [1, 4096]"""
        if activations.ndim == 2 and activations.shape[0] == 7:
            x = activations.mean(dim=0, keepdim=True)
        elif activations.ndim == 2 and activations.shape[0] == 1:
            x = activations
        else:
            raise ValueError(f"Expected activations [7, d] or [1, d], got {activations.shape}")

        device = next(self.sae.parameters()).device
        features = self.sae.encode(x.to(device=device, dtype=torch.float32))
        return features.squeeze(0).detach().cpu().numpy()

    def train_lr_monitor(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
        self.lr_model.fit(X_train, y_train)

    def predict_score(self, activations: torch.Tensor, method: str = "lr") -> float:
        features = self.extract_features(activations)
        if method == "lr":
            if self.lr_model is None:
                raise RuntimeError("LR monitor is not trained")
            return float(self.lr_model.predict_proba(features.reshape(1, -1))[0, 1])
        if method == "threshold":
            if self.feature_weights is None:
                raise RuntimeError("feature_weights is required for threshold monitor")
            return float((features * self.feature_weights).sum())
        raise ValueError(f"Unknown method: {method}")

    def predict_binary(self, activations: torch.Tensor, method: str = "lr", threshold: float | None = None) -> int:
        thr = self.threshold if threshold is None else threshold
        score = self.predict_score(activations, method=method)
        return int(score >= thr)

    @staticmethod
    def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> MonitorMetrics:
        y_pred = (y_score >= threshold).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        return MonitorMetrics(
            auroc=float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.5,
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            cost_weighted_f1=float(cost_weighted_f1(y_true, y_pred, fn_weight=10.0)),
            pr_auc=float(pr_auc),
        )
