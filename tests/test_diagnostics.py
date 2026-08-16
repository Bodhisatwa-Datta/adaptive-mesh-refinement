import numpy as np
import pytest

from amr.diagnostics.errors import error_norms


def test_error_norm_definitions() -> None:
    errors = error_norms(np.array([1.0, -1.0]), np.zeros(2))
    assert errors.l1 == pytest.approx(1.0)
    assert errors.l2 == pytest.approx(1.0)
    assert errors.linf == pytest.approx(1.0)

