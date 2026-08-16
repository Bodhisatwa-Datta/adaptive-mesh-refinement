import numpy as np

from amr.numerics.boundary_conditions import fill_periodic_ghost_cells


def test_periodic_ghost_cells_wrap_both_ends() -> None:
    result = fill_periodic_ghost_cells([0.0, 1.0, 2.0, 3.0], n_ghost=2)
    np.testing.assert_array_equal(result, [2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0])

