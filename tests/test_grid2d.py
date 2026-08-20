import numpy as np
import pytest

from amr.grid.grid2d import UniformGrid2D


def test_cartesian_coordinates_and_array_convention() -> None:
    grid = UniformGrid2D(-1.0, 1.0, 4, 2.0, 3.0, 2)
    x, y = grid.cell_centres

    assert grid.shape == (2, 4)
    assert grid.dx == pytest.approx(0.5)
    assert grid.dy == pytest.approx(0.5)
    assert grid.cell_area == pytest.approx(0.25)
    np.testing.assert_allclose(grid.x_centres, [-0.75, -0.25, 0.25, 0.75])
    np.testing.assert_allclose(grid.y_centres, [2.25, 2.75])
    np.testing.assert_allclose(grid.x_edges, [-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(grid.y_edges, [2.0, 2.5, 3.0])
    np.testing.assert_allclose(x[0], grid.x_centres)
    np.testing.assert_allclose(y[:, 0], grid.y_centres)


@pytest.mark.parametrize("shape", [(1, 4), (4, 1)])
def test_grid_requires_at_least_two_cells_per_direction(
    shape: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        UniformGrid2D(0.0, 1.0, shape[0], 0.0, 1.0, shape[1])


def test_grid_rejects_field_with_transposed_shape() -> None:
    grid = UniformGrid2D(0.0, 1.0, 6, 0.0, 1.0, 4)
    with pytest.raises(ValueError, match="Expected field shape"):
        grid.validate_field(np.zeros((6, 4)))
