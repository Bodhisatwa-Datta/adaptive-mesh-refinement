import numpy as np

from amr.refinement.criteria import (
    bounding_box_from_flags_2d,
    boxes_from_flags_2d,
    buffer_flags_2d,
    flag_gradient_2d,
    gradient_indicator_2d,
)


def test_uniform_2d_state_has_zero_gradient() -> None:
    indicator = gradient_indicator_2d(np.full((6, 8), 3.0), dx=0.2, dy=0.1)
    np.testing.assert_array_equal(indicator, np.zeros((6, 8)))


def test_linear_field_has_expected_nonperiodic_gradient() -> None:
    x = (np.arange(8) + 0.5) * 0.25
    y = (np.arange(6) + 0.5) * 0.5
    field = 2.0 * x[None, :] - 3.0 * y[:, None]
    indicator = gradient_indicator_2d(
        field, dx=0.25, dy=0.5, periodic=False
    )
    np.testing.assert_allclose(indicator, np.hypot(2.0, 3.0))


def test_periodic_2d_buffer_wraps_across_both_axes() -> None:
    flags = np.zeros((5, 6), dtype=bool)
    flags[0, 0] = True
    expanded = buffer_flags_2d(flags, 1, periodic=True)
    expected_y = [0, 1, 4]
    expected_x = [0, 1, 5]
    assert np.count_nonzero(expanded) == 9
    assert np.all(expanded[np.ix_(expected_y, expected_x)])


def test_gradient_flags_form_a_half_open_bounding_box() -> None:
    values = np.zeros((10, 12))
    values[3:7, 4:9] = 1.0
    flags = flag_gradient_2d(
        values, dx=1.0, dy=1.0, threshold=0.1, n_buffer=1, periodic=False
    )
    assert bounding_box_from_flags_2d(flags) == (2, 11, 1, 9)


def test_empty_flags_have_no_bounding_box() -> None:
    assert bounding_box_from_flags_2d(np.zeros((4, 5), dtype=bool)) is None


def test_separated_components_produce_multiple_boxes() -> None:
    flags = np.zeros((12, 16), dtype=bool)
    flags[1:4, 2:5] = True
    flags[7:11, 10:15] = True
    assert boxes_from_flags_2d(flags) == [(2, 5, 1, 4), (10, 15, 7, 11)]


def test_nearby_component_boxes_can_be_merged() -> None:
    flags = np.zeros((10, 12), dtype=bool)
    flags[2:4, 2:4] = True
    flags[2:4, 6:8] = True
    assert boxes_from_flags_2d(flags, merge_gap=1) == [
        (2, 4, 2, 4),
        (6, 8, 2, 4),
    ]
    assert boxes_from_flags_2d(flags, merge_gap=2) == [(2, 8, 2, 4)]


def test_periodic_edge_components_remain_separate_boxes() -> None:
    flags = np.zeros((8, 10), dtype=bool)
    flags[3:5, :2] = True
    flags[3:5, -2:] = True
    assert boxes_from_flags_2d(flags) == [(0, 2, 3, 5), (8, 10, 3, 5)]
