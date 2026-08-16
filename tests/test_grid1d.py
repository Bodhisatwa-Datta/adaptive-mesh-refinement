import numpy as np
import pytest

from amr.grid.grid1d import UniformGrid1D


def test_cell_coordinates_are_cell_centred() -> None:
    grid = UniformGrid1D(-1.0, 1.0, 4)
    np.testing.assert_allclose(grid.cell_centres, [-0.75, -0.25, 0.25, 0.75])
    np.testing.assert_allclose(grid.cell_edges, [-1.0, -0.5, 0.0, 0.5, 1.0])
    assert grid.dx == pytest.approx(0.5)


@pytest.mark.parametrize("bounds", [(1.0, 1.0), (2.0, 1.0)])
def test_invalid_domain_is_rejected(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        UniformGrid1D(*bounds, n_cells=10)

