from __future__ import annotations

import pytest


@pytest.mark.unit
def test_package_imports_and_exposes_version() -> None:
    import georl
    import georl.data
    import georl.evaluation
    import georl.hsa
    import georl.losses
    import georl.metrics
    import georl.models
    import georl.rl
    import georl.training
    import georl.utils

    assert isinstance(georl.__version__, str)
    assert georl.__version__.count(".") >= 1
