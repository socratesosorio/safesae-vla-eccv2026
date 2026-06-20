import os

import pytest

pytestmark = pytest.mark.cloud


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION", "0") != "1", reason="Set RUN_INTEGRATION=1 to run")
def test_modal_entrypoints_present():
    import modal_app

    assert hasattr(modal_app, "collect")
    assert hasattr(modal_app, "train_sae")
    assert hasattr(modal_app, "analyze")
    assert hasattr(modal_app, "evaluate")
