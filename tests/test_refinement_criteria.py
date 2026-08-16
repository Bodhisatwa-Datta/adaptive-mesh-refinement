import numpy as np

from amr.refinement.criteria import (
    buffer_flags,
    flag_gradient,
    gradient_indicator,
    regions_from_flags,
)


def test_uniform_state_has_zero_gradient_indicator() -> None:
    indicator = gradient_indicator(np.full(12, 7.0), dx=0.1)
    np.testing.assert_array_equal(indicator, np.zeros(12))


def test_periodic_gradient_flags_both_sides_of_discontinuities() -> None:
    values = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    flags = flag_gradient(values, dx=0.125, threshold=1.0, periodic=True)
    np.testing.assert_array_equal(np.flatnonzero(flags), [0, 3, 4, 7])


def test_buffer_expands_and_wraps_periodically() -> None:
    flags = np.array([True, False, False, False, False, False])
    expanded = buffer_flags(flags, n_buffer=2, periodic=True)
    np.testing.assert_array_equal(np.flatnonzero(expanded), [0, 1, 2, 4, 5])


def test_nonperiodic_buffer_stops_at_domain_edge() -> None:
    flags = np.array([True, False, False, False, False])
    expanded = buffer_flags(flags, n_buffer=2, periodic=False)
    np.testing.assert_array_equal(np.flatnonzero(expanded), [0, 1, 2])


def test_normalized_indicator_is_scale_independent_away_from_zero() -> None:
    values = np.array([1.0, 1.2, 1.6, 1.3, 1.1])
    original = gradient_indicator(values, dx=0.2, normalized=True, epsilon=1.0e-30)
    scaled = gradient_indicator(10.0 * values, dx=0.2, normalized=True, epsilon=1.0e-30)
    np.testing.assert_allclose(original, scaled)


def test_regions_are_half_open_and_small_gaps_can_be_merged() -> None:
    flags = np.array([False, True, True, False, True, False, False, True])
    assert regions_from_flags(flags) == [(1, 3), (4, 5), (7, 8)]
    assert regions_from_flags(flags, merge_gap=1) == [(1, 5), (7, 8)]

