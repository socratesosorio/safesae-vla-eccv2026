import numpy as np
import pytest
import torch

pytest.importorskip("statsmodels")

from src.analysis.differential_activation import DifferentialActivationAnalyzer
from src.sae.model import BatchTopKSAE


class DummyDataset:
    def __init__(self):
        self.metadata = []
        self.items = []
        for i in range(6):
            act = torch.randn(5, 7, 8)
            has_viol = bool(i % 2)
            viol_count = {"collision": 1 if has_viol else 0, "excessive_force": 0, "boundary_violation": 0, "high_approach_speed": 0, "object_drop": 0}
            self.items.append({"activations_layer16": act})
            self.metadata.append({"has_violations": has_viol, "violation_counts": viol_count})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def test_differential_analysis_runs():
    sae = BatchTopKSAE(d_in=8, d_sae=16, k=4)
    analyzer = DifferentialActivationAnalyzer(sae=sae, config={})
    results = analyzer.run_layer_analysis(DummyDataset(), layer=16)
    df = results["overall"]
    assert not df.empty
    assert {"feature_idx", "adjusted_p", "composite_score"}.issubset(set(df.columns))
