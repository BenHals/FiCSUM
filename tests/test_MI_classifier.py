from ConceptFingerprint.Classifier.meta_info_classifier import (
    window_to_timeseries,
    update_timeseries
)
from ConceptFingerprint.Classifier.fingerprint import (Fingerprint)
from ConceptFingerprint.Classifier.normalizer import Normalizer
import random, time, statistics

import numpy as np




def test_normalizer_add_stats():
    normalizer = Normalizer(fingerprint_constructor=Fingerprint)
    stats = {
        "Feature": {"Mean": 100, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 50},
    }
    normalizer.add_stats(stats)
    for s in stats:
        for f in stats[s]:
            norm_min, norm_max = normalizer.data_ranges[s][f]
            assert norm_min == stats[s][f]
            assert norm_max == stats[s][f]
            flat_index = normalizer.total_indexes[s][f]
            flat_max, flat_min = normalizer.total_flat_max[flat_index], normalizer.total_flat_min[flat_index]
            assert flat_min == stats[s][f]
            assert flat_max == stats[s][f]
            ignore_index = normalizer.ignore_indexes[s][f]
            ignore_max, ignore_min = normalizer.ignore_flat_max[ignore_index], normalizer.ignore_flat_min[ignore_index]
            assert ignore_min == stats[s][f]
            assert ignore_max == stats[s][f]

            distribution_info = normalizer.data_stdev[s][f]
            assert distribution_info["seen"] == 1
            assert distribution_info["value"] == stats[s][f]
            assert distribution_info["stdev"] == 0

    stats_2 = {
        "Feature": {"Mean": 150, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 25},
    }
    normalizer.add_stats(stats_2)
    for s in stats:
        for f in stats[s]:
            norm_min, norm_max = normalizer.data_ranges[s][f]
            assert norm_min == min(stats[s][f], stats_2[s][f])
            assert norm_max == max(stats[s][f], stats_2[s][f])
            flat_index = normalizer.total_indexes[s][f]
            flat_max, flat_min = normalizer.total_flat_max[flat_index], normalizer.total_flat_min[flat_index]
            assert flat_min == min(stats[s][f], stats_2[s][f])
            assert flat_max == max(stats[s][f], stats_2[s][f])
            ignore_index = normalizer.ignore_indexes[s][f]
            ignore_max, ignore_min = normalizer.ignore_flat_max[ignore_index], normalizer.ignore_flat_min[ignore_index]
            assert ignore_min == min(stats[s][f], stats_2[s][f])
            assert ignore_max == max(stats[s][f], stats_2[s][f])

            distribution_info = normalizer.data_stdev[s][f]
            assert distribution_info["seen"] == 2
            assert distribution_info["value"] == statistics.mean((stats[s][f], stats_2[s][f]))
            assert distribution_info["stdev"] == statistics.stdev((stats[s][f], stats_2[s][f]))
def test_normalizer_ignore_source_add_stats():
    normalizer = Normalizer(fingerprint_constructor=Fingerprint, ignore_sources=["Labels"])
    assert normalizer.ignore_features == []
    assert normalizer.ignore_sources == ["Labels"]
    stats = {
        "Feature": {"Mean": 100, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 50},
    }
    normalizer.add_stats(stats)
    for s in stats:
        for f in stats[s]:
            norm_min, norm_max = normalizer.data_ranges[s][f]
            assert norm_min == stats[s][f]
            assert norm_max == stats[s][f]
            flat_index = normalizer.total_indexes[s][f]
            flat_max, flat_min = normalizer.total_flat_max[flat_index], normalizer.total_flat_min[flat_index]
            assert flat_min == stats[s][f]
            assert flat_max == stats[s][f]
            if s != "Labels":
                ignore_index = normalizer.ignore_indexes[s][f]
                ignore_max, ignore_min = normalizer.ignore_flat_max[ignore_index], normalizer.ignore_flat_min[ignore_index]
                assert ignore_min == stats[s][f]
                assert ignore_max == stats[s][f]
            else:
                assert s not in normalizer.ignore_indexes

            distribution_info = normalizer.data_stdev[s][f]
            assert distribution_info["seen"] == 1
            assert distribution_info["value"] == stats[s][f]
            assert distribution_info["stdev"] == 0

    stats_2 = {
        "Feature": {"Mean": 150, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 25},
    }
    normalizer.add_stats(stats_2)
    for s in stats:
        for f in stats[s]:
            norm_min, norm_max = normalizer.data_ranges[s][f]
            assert norm_min == min(stats[s][f], stats_2[s][f])
            assert norm_max == max(stats[s][f], stats_2[s][f])
            flat_index = normalizer.total_indexes[s][f]
            flat_max, flat_min = normalizer.total_flat_max[flat_index], normalizer.total_flat_min[flat_index]
            assert flat_min == min(stats[s][f], stats_2[s][f])
            assert flat_max == max(stats[s][f], stats_2[s][f])
            if s != "Labels":
                ignore_index = normalizer.ignore_indexes[s][f]
                ignore_max, ignore_min = normalizer.ignore_flat_max[ignore_index], normalizer.ignore_flat_min[ignore_index]
                assert ignore_min == min(stats[s][f], stats_2[s][f])
                assert ignore_max == max(stats[s][f], stats_2[s][f])
            else:
                assert s not in normalizer.ignore_indexes

            distribution_info = normalizer.data_stdev[s][f]
            assert distribution_info["seen"] == 2
            assert distribution_info["value"] == statistics.mean((stats[s][f], stats_2[s][f]))
            assert distribution_info["stdev"] == statistics.stdev((stats[s][f], stats_2[s][f]))
def test_normalizer_ignore_feature_add_stats():
    normalizer = Normalizer(fingerprint_constructor=Fingerprint, ignore_features=["Mean"])
    assert normalizer.ignore_features == ["Mean"]
    assert normalizer.ignore_sources == []
    stats = {
        "Feature": {"Mean": 100, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 50},
    }
    normalizer.add_stats(stats)
    for s in stats:
        for f in stats[s]:
            norm_min, norm_max = normalizer.data_ranges[s][f]
            assert norm_min == stats[s][f]
            assert norm_max == stats[s][f]
            flat_index = normalizer.total_indexes[s][f]
            flat_max, flat_min = normalizer.total_flat_max[flat_index], normalizer.total_flat_min[flat_index]
            assert flat_min == stats[s][f]
            assert flat_max == stats[s][f]
            if f != "Mean":
                ignore_index = normalizer.ignore_indexes[s][f]
                ignore_max, ignore_min = normalizer.ignore_flat_max[ignore_index], normalizer.ignore_flat_min[ignore_index]
                assert ignore_min == stats[s][f]
                assert ignore_max == stats[s][f]
            else:
                assert f not in normalizer.ignore_indexes[s]

            distribution_info = normalizer.data_stdev[s][f]
            assert distribution_info["seen"] == 1
            assert distribution_info["value"] == stats[s][f]
            assert distribution_info["stdev"] == 0

    stats_2 = {
        "Feature": {"Mean": 150, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 25},
    }
    normalizer.add_stats(stats_2)
    for s in stats:
        for f in stats[s]:
            norm_min, norm_max = normalizer.data_ranges[s][f]
            assert norm_min == min(stats[s][f], stats_2[s][f])
            assert norm_max == max(stats[s][f], stats_2[s][f])
            flat_index = normalizer.total_indexes[s][f]
            flat_max, flat_min = normalizer.total_flat_max[flat_index], normalizer.total_flat_min[flat_index]
            assert flat_min == min(stats[s][f], stats_2[s][f])
            assert flat_max == max(stats[s][f], stats_2[s][f])
            if f != "Mean":
                ignore_index = normalizer.ignore_indexes[s][f]
                ignore_max, ignore_min = normalizer.ignore_flat_max[ignore_index], normalizer.ignore_flat_min[ignore_index]
                assert ignore_min == min(stats[s][f], stats_2[s][f])
                assert ignore_max == max(stats[s][f], stats_2[s][f])
            else:
                assert f not in normalizer.ignore_indexes[s]

            distribution_info = normalizer.data_stdev[s][f]
            assert distribution_info["seen"] == 2
            assert distribution_info["value"] == statistics.mean((stats[s][f], stats_2[s][f]))
            assert distribution_info["stdev"] == statistics.stdev((stats[s][f], stats_2[s][f]))

def test_normalizer_get_normed_value():
    normalizer = Normalizer(fingerprint_constructor=Fingerprint)
    stats = {
        "Feature": {"Mean": 100, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 50},
    }
    normalizer.add_stats(stats)

    stats_2 = {
        "Feature": {"Mean": 150, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 25},
    }
    normalizer.add_stats(stats_2)

    for s in stats_2:
        for f in stats_2[s]:
            normed_val = normalizer.get_normed_value(s, f, stats_2[s][f])
            min_seen = min(stats[s][f], stats_2[s][f])
            max_seen = max(stats[s][f], stats_2[s][f])
            seen_range = (max_seen - min_seen)
            # If we have no range, we set the normed value to be 0.5.
            # This is reduce jumps in value when we do see a range,
            # as 0.5 is the closest to every point in 0-1.
            if seen_range == 0:
                assert normed_val == 0.5
            else:
                assert normed_val == (stats_2[s][f] - min_seen) / seen_range

def test_normalizer_get_flat_vec():
    """ After initializing a normalizer,
    We should be able to call get_flat_vector
    on a new fingerprint to get a flattened version
    in a canonical order.
    Should return both a normalized and unnormalized version.
    Normalization should be based on the min-max seen so far.
    """
    normalizer = Normalizer(fingerprint_constructor=Fingerprint)
    stats = {
        "Feature": {"Mean": 100, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 50},
    }

    normalizer.add_stats(stats)
    stats_2 = {
        "Feature": {"Mean": 150, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 25},
    }

    normalizer.add_stats(stats_2)

    # Expected array should be in the order
    # given by normalizer.source_order and feature_order
    expected_unnormalized_array = []
    expected_normalized_array = []
    for s in normalizer.source_order:
        for f in normalizer.feature_order:
            expected_unnormalized_array.append(stats_2[s][f])
            seen_values = [stats_2[s][f], stats[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            seen_range = (max_seen - min_seen)
            normalized_expected_value = ((stats_2[s][f] - min_seen) / seen_range) if seen_range > 0 else 0.5
            expected_normalized_array.append(normalized_expected_value)
    expected_return = (np.array(expected_normalized_array), np.array(expected_unnormalized_array))
    np.testing.assert_equal(expected_return, normalizer.get_flat_vector(stats_2, use_full=True))

    expected_unnormalized_array = []
    expected_normalized_array = []
    for s in normalizer.source_order:
        for f in normalizer.feature_order:
            expected_unnormalized_array.append(stats[s][f])
            seen_values = [stats_2[s][f], stats[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            seen_range = (max_seen - min_seen)
            normalized_expected_value = ((stats[s][f] - min_seen) / seen_range) if seen_range > 0 else 0.5
            expected_normalized_array.append(normalized_expected_value)
    expected_return = (np.array(expected_normalized_array), np.array(expected_unnormalized_array))
    np.testing.assert_equal(expected_return, normalizer.get_flat_vector(stats, use_full=True))

def test_normalizer_ignore_source_get_flat_vec():
    """ After initializing a normalizer,
    We should be able to call get_flat_vector
    on a new fingerprint to get a flattened version
    in a canonical order.
    Should return both a normalized and unnormalized version.
    Normalization should be based on the min-max seen so far.
    Testing ignore.
    """
    normalizer = Normalizer(fingerprint_constructor=Fingerprint, ignore_sources=["Feature"])
    assert normalizer.ignore_sources == ["Feature"]
    stats = {
        "Feature": {"Mean": 100, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 50},
    }

    normalizer.add_stats(stats)
    stats_2 = {
        "Feature": {"Mean": 150, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 25},
    }

    normalizer.add_stats(stats_2)

    # Expected array should be in the order
    # given by normalizer.source_order and feature_order
    # If use_full is True, should include ignored source
    expected_unnormalized_array = []
    expected_normalized_array = []
    for s in normalizer.source_order:
        for f in normalizer.feature_order:
            expected_unnormalized_array.append(stats_2[s][f])
            seen_values = [stats_2[s][f], stats[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            seen_range = (max_seen - min_seen)
            normalized_expected_value = ((stats_2[s][f] - min_seen) / seen_range) if seen_range > 0 else 0.5
            expected_normalized_array.append(normalized_expected_value)
    expected_return = (np.array(expected_normalized_array), np.array(expected_unnormalized_array))
    np.testing.assert_equal(expected_return, normalizer.get_flat_vector(stats_2, use_full=True))

    expected_unnormalized_array = []
    expected_normalized_array = []
    for s in normalizer.source_order:
        for f in normalizer.feature_order:
            expected_unnormalized_array.append(stats[s][f])
            seen_values = [stats_2[s][f], stats[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            seen_range = (max_seen - min_seen)
            normalized_expected_value = ((stats[s][f] - min_seen) / seen_range) if seen_range > 0 else 0.5
            expected_normalized_array.append(normalized_expected_value)
    expected_return = (np.array(expected_normalized_array), np.array(expected_unnormalized_array))
    np.testing.assert_equal(expected_return, normalizer.get_flat_vector(stats, use_full=True))
    # If use_full is False, should NOT include ignored source
    expected_unnormalized_array = []
    expected_normalized_array = []
    for s in normalizer.source_order:
        if s == "Feature": continue
        for f in normalizer.feature_order:
            expected_unnormalized_array.append(stats_2[s][f])
            seen_values = [stats_2[s][f], stats[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            seen_range = (max_seen - min_seen)
            normalized_expected_value = ((stats_2[s][f] - min_seen) / seen_range) if seen_range > 0 else 0.5
            expected_normalized_array.append(normalized_expected_value)
    expected_return = (np.array(expected_normalized_array), np.array(expected_unnormalized_array))
    np.testing.assert_equal(expected_return, normalizer.get_flat_vector(stats_2, use_full=False))

    expected_unnormalized_array = []
    expected_normalized_array = []
    for s in normalizer.source_order:
        if s == "Feature": continue
        for f in normalizer.feature_order:
            expected_unnormalized_array.append(stats[s][f])
            seen_values = [stats_2[s][f], stats[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            seen_range = (max_seen - min_seen)
            normalized_expected_value = ((stats[s][f] - min_seen) / seen_range) if seen_range > 0 else 0.5
            expected_normalized_array.append(normalized_expected_value)
    expected_return = (np.array(expected_normalized_array), np.array(expected_unnormalized_array))
    np.testing.assert_equal(expected_return, normalizer.get_flat_vector(stats, use_full=False))

def test_normalizer_ignore_feature_get_flat_vec():
    """ After initializing a normalizer,
    We should be able to call get_flat_vector
    on a new fingerprint to get a flattened version
    in a canonical order.
    Should return both a normalized and unnormalized version.
    Normalization should be based on the min-max seen so far.
    Testing ignore.
    """
    normalizer = Normalizer(fingerprint_constructor=Fingerprint, ignore_features=["FI"])
    assert normalizer.ignore_sources == []
    assert normalizer.ignore_features == ["FI"]
    stats = {
        "Feature": {"Mean": 100, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 50},
    }

    normalizer.add_stats(stats)
    stats_2 = {
        "Feature": {"Mean": 150, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 25},
    }

    normalizer.add_stats(stats_2)

    # Expected array should be in the order
    # given by normalizer.source_order and feature_order
    # If use_full is True, should include ignored source
    expected_unnormalized_array = []
    expected_normalized_array = []
    for s in normalizer.source_order:
        for f in normalizer.feature_order:
            expected_unnormalized_array.append(stats_2[s][f])
            seen_values = [stats_2[s][f], stats[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            seen_range = (max_seen - min_seen)
            normalized_expected_value = ((stats_2[s][f] - min_seen) / seen_range) if seen_range > 0 else 0.5
            expected_normalized_array.append(normalized_expected_value)
    expected_return = (np.array(expected_normalized_array), np.array(expected_unnormalized_array))
    np.testing.assert_equal(expected_return, normalizer.get_flat_vector(stats_2, use_full=True))

    expected_unnormalized_array = []
    expected_normalized_array = []
    for s in normalizer.source_order:
        for f in normalizer.feature_order:
            expected_unnormalized_array.append(stats[s][f])
            seen_values = [stats_2[s][f], stats[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            seen_range = (max_seen - min_seen)
            normalized_expected_value = ((stats[s][f] - min_seen) / seen_range) if seen_range > 0 else 0.5
            expected_normalized_array.append(normalized_expected_value)
    expected_return = (np.array(expected_normalized_array), np.array(expected_unnormalized_array))
    np.testing.assert_equal(expected_return, normalizer.get_flat_vector(stats, use_full=True))
    # If use_full is False, should NOT include ignored source
    expected_unnormalized_array = []
    expected_normalized_array = []
    for s in normalizer.source_order:
        for f in normalizer.feature_order:
            if f == "FI": continue
            expected_unnormalized_array.append(stats_2[s][f])
            seen_values = [stats_2[s][f], stats[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            seen_range = (max_seen - min_seen)
            normalized_expected_value = ((stats_2[s][f] - min_seen) / seen_range) if seen_range > 0 else 0.5
            expected_normalized_array.append(normalized_expected_value)
    expected_return = (np.array(expected_normalized_array), np.array(expected_unnormalized_array))
    np.testing.assert_equal(expected_return, normalizer.get_flat_vector(stats_2, use_full=False))

    expected_unnormalized_array = []
    expected_normalized_array = []
    for s in normalizer.source_order:
        for f in normalizer.feature_order:
            if f == "FI": continue
            expected_unnormalized_array.append(stats[s][f])
            seen_values = [stats_2[s][f], stats[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            seen_range = (max_seen - min_seen)
            normalized_expected_value = ((stats[s][f] - min_seen) / seen_range) if seen_range > 0 else 0.5
            expected_normalized_array.append(normalized_expected_value)
    expected_return = (np.array(expected_normalized_array), np.array(expected_unnormalized_array))
    np.testing.assert_equal(expected_return, normalizer.get_flat_vector(stats, use_full=False))


def test_normalizer_get_flat_labels():
    normalizer = Normalizer(fingerprint_constructor=Fingerprint)
    stats = {
        "Feature": {"Mean": 100, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 50},
    }

    normalizer.add_stats(stats)
    stats_2 = {
        "Feature": {"Mean": 150, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 25},
    }

    normalizer.add_stats(stats_2)
    expected_labels = []
    for s in normalizer.source_order:
        for f in normalizer.feature_order:
            expected_labels.append(f"{s}-{f}")
    expected_return = (np.array(expected_labels))
    np.testing.assert_equal(expected_return, normalizer.get_flat_vector_labels(stats_2, use_full=True))

    # print(normalizer.get_flat_vector_labels(stats_2, use_full=True))

def test_normalizer_get_flat_ranges():
    normalizer = Normalizer(fingerprint_constructor=Fingerprint)
    stats = {
        "Feature": {"Mean": 100, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 50},
    }

    normalizer.add_stats(stats)
    stats_2 = {
        "Feature": {"Mean": 150, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 25},
    }

    normalizer.add_stats(stats_2)
    expected_ranges = []
    for s in normalizer.source_order:
        for f in normalizer.feature_order:
            seen_values = [stats_2[s][f], stats[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            expected_ranges.append(f"{min_seen}:{max_seen}")
    expected_return = (np.array(expected_ranges))
    np.testing.assert_equal(expected_return, normalizer.get_flat_vector_ranges(stats_2, use_full=True))



def test_normalizer_norm_vector():
    normalizer = Normalizer(fingerprint_constructor=Fingerprint, ignore_sources=["Labels"])
    normalizer2 = Normalizer(fingerprint_constructor=Fingerprint, ignore_features=["FI"])
    stats = {
        "Feature": {"Mean": 100, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 50},
    }

    normalizer.add_stats(stats)
    normalizer2.add_stats(stats)
    stats_2 = {
        "Feature": {"Mean": 150, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 25},
    }

    normalizer.add_stats(stats_2)
    normalizer2.add_stats(stats_2)
    stats_3 = {
        "Feature": {"Mean": 125, "FI": 50},
        "Labels": {"Mean": 0.5, "FI": 10},
    }
    normalizer.add_stats(stats_3)
    normalizer2.add_stats(stats_3)

    full_1_normed, full_1_no_normed = normalizer.get_flat_vector(stats_3, use_full=True)
    ignore_1_normed, ignore_1_no_normed = normalizer.get_flat_vector(stats_3, use_full=False)
    full_2_normed, full_2_no_normed = normalizer2.get_flat_vector(stats_3, use_full=True)
    ignore_2_normed, ignore_2_no_normed = normalizer2.get_flat_vector(stats_3, use_full=False)


    expected_unnormalized_array = []
    expected_normalized_array = []
    for s in normalizer.source_order:
        for f in normalizer.feature_order:
            expected_unnormalized_array.append(stats_3[s][f])
            seen_values = [stats[s][f], stats_2[s][f], stats_3[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            seen_range = (max_seen - min_seen)
            normalized_expected_value = ((stats_3[s][f] - min_seen) / seen_range) if seen_range > 0 else 0.5
            expected_normalized_array.append(normalized_expected_value)
    expected_return = (np.array(expected_normalized_array), np.array(expected_unnormalized_array))
    np.testing.assert_equal(expected_return[0], full_1_normed)
    np.testing.assert_equal(expected_return[1], full_1_no_normed)
    np.testing.assert_equal(expected_return[0], full_2_normed)
    np.testing.assert_equal(expected_return[1], full_2_no_normed)
    expected_unnormalized_array = []
    expected_normalized_array = []
    for s in normalizer.source_order:
        if s == "Labels": continue
        for f in normalizer.feature_order:
            expected_unnormalized_array.append(stats_3[s][f])
            seen_values = [stats[s][f], stats_2[s][f], stats_3[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            seen_range = (max_seen - min_seen)
            normalized_expected_value = ((stats_3[s][f] - min_seen) / seen_range) if seen_range > 0 else 0.5
            expected_normalized_array.append(normalized_expected_value)
    expected_return = (np.array(expected_normalized_array), np.array(expected_unnormalized_array))
    np.testing.assert_equal(expected_return[0], ignore_1_normed)
    np.testing.assert_equal(expected_return[1], ignore_1_no_normed)
    expected_unnormalized_array = []
    expected_normalized_array = []
    for s in normalizer.source_order:
        for f in normalizer.feature_order:
            if f == "FI": continue
            expected_unnormalized_array.append(stats_3[s][f])
            seen_values = [stats[s][f], stats_2[s][f], stats_3[s][f]]
            max_seen = max(seen_values)
            min_seen = min(seen_values)
            seen_range = (max_seen - min_seen)
            normalized_expected_value = ((stats_3[s][f] - min_seen) / seen_range) if seen_range > 0 else 0.5
            expected_normalized_array.append(normalized_expected_value)
    expected_return = (np.array(expected_normalized_array), np.array(expected_unnormalized_array))
    np.testing.assert_equal(expected_return[0], ignore_2_normed)
    np.testing.assert_equal(expected_return[1], ignore_2_no_normed)

def test_normalizer_order_dict():
    """ Order dict should put a nested
    dict structed as [source][feature]
    into the canonical flat vector.
    Also returns labels and ranges.
    """
    normalizer_full = Normalizer(fingerprint_constructor=Fingerprint)
    normalizer = Normalizer(fingerprint_constructor=Fingerprint, ignore_sources=["Labels"])
    normalizer2 = Normalizer(fingerprint_constructor=Fingerprint, ignore_features=["FI"])
    stats = {
        "Feature": {"Mean": 100, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 50},
    }

    normalizer_full.add_stats(stats)
    normalizer.add_stats(stats)
    normalizer2.add_stats(stats)
    stats_2 = {
        "Feature": {"Mean": 150, "FI": 25},
        "Labels": {"Mean": 0.5, "FI": 25},
    }

    normalizer_full.add_stats(stats_2)
    normalizer.add_stats(stats_2)
    normalizer2.add_stats(stats_2)
    stats_3 = {
        "Feature": {"Mean": 125, "FI": 50},
        "Labels": {"Mean": 0.5, "FI": 10},
    }
    normalizer_full.add_stats(stats_3)
    normalizer.add_stats(stats_3)
    normalizer2.add_stats(stats_3)

    weights = {
        "Feature": {"Mean": 0.5, "FI": 0.1},
        "Labels": {"Mean": 0.75, "FI": 0.2},
    }

    N1_ordered = []
    N1_labels= []
    N1_ranges= []
    N2_ordered = []
    N2_labels= []
    N2_ranges= []
    N3_ordered = []
    N3_labels= []
    N3_ranges= []
    for s in normalizer.source_order:
        for f in normalizer.feature_order:
            min_seen = min(stats[s][f], stats_2[s][f], stats_3[s][f])
            max_seen = max(stats[s][f], stats_2[s][f], stats_3[s][f])
            N1_ordered.append(weights[s][f])
            N1_labels.append(f"{s}-{f}")
            N1_ranges.append(f"{min_seen}:{max_seen}")
            if s != "Labels":
                N2_ordered.append(weights[s][f])
                N2_labels.append(f"{s}-{f}")
                N2_ranges.append(f"{min_seen}:{max_seen}")
            if f != "FI":
                N3_ordered.append(weights[s][f])
                N3_labels.append(f"{s}-{f}")
                N3_ranges.append(f"{min_seen}:{max_seen}")
    np.testing.assert_equal([np.array(N1_ordered), np.array(N1_labels), np.array(N1_ranges)], normalizer_full.order_dict(weights, normalize=False))
    np.testing.assert_equal([np.array(N2_ordered), np.array(N2_labels), np.array(N2_ranges)], normalizer.order_dict(weights, normalize=False))
    np.testing.assert_equal([np.array(N3_ordered), np.array(N3_labels), np.array(N3_ranges)], normalizer2.order_dict(weights, normalize=False))


def test_window_to_timeseries():
    """ Function to turn a sequence of observations
    of the format, X, y, p, is_correct,
    where X is a vector of features,
    into the arrays
    [[x11, x12, ...], [x21, x22, ...], ...], [y1, y2, ...], [p1, p2, ...], [p1==y1, p2==y2, ...], [e2-e1, e3-2, ...]

    """
    window = [
        ([1, 2, 3], 1, 1, 1),
        ([1, 2, 4], 0, 1, 0),
        ([3, 2, 3], 1, 0, 0),
        ([5, 1, 6], 0, 0, 1)
    ]
    timeseries = window_to_timeseries([window, "n"])
    expected_timeseries = ([[1, 1, 3, 5], [2, 2, 2, 1], [3, 4, 3, 6]], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1], [1])
    np.testing.assert_equal(expected_timeseries, timeseries)

def test_window_to_timeseries_update():
    """ update_timeseries is a quick
    method to update some elements from a previous
    timeseries. Should behave the same as window_to_timeseries
    but faster. The first error_distance value may be different
    over the split, but this is acceptable.

    """
    window = [
        ([1, 2, 3], 1, 1, 1),
        ([1, 2, 4], 0, 1, 0),
        ([3, 2, 3], 1, 0, 0),
        ([5, 1, 6], 0, 0, 1)
    ]
    timeseries = window_to_timeseries([window, "n"])
    window_2 = [
        ([3, 2, 3], 1, 0, 0),
        ([5, 1, 6], 0, 0, 1),
        ([0, 1, 3], 0, 1, 0),
        ([6, 2, 4], 1, 1, 1)
    ]
    timeseries_2 = window_to_timeseries([window_2, "n"])
    timeseries_2_update = update_timeseries(timeseries, [window_2, "n"], 4, 2)
    expected_timeseries = ([[3, 5, 0, 6], [2, 1, 1, 2], [3, 6, 3, 4]], [1, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [2])
    np.testing.assert_equal(expected_timeseries, timeseries_2)
    np.testing.assert_equal(expected_timeseries, timeseries_2_update)





def test_window_to_timeseries_update_correct_no_inplace():
    num_features = 10
    observations = []
    for i in range(1000):
        features = []
        for f in range(num_features):
            features.append(random.random())
        label = random.randint(0, 2)
        p = random.randint(0, 2)
        e = label == p
        observations.append((features, label, p, e))
    
    window_size = 250
    standard_timeseries = []
    standard_start = time.time()
    for e in range(250, len(observations), 50):
        window = observations[e-window_size:e]
        timeseries = window_to_timeseries([window, "n"])
        standard_timeseries.append(timeseries)
    standard_end = time.time()

    update_timeseries_record = []
    last_window = None
    last_ob = None
    update_start = time.time()
    for e in range(250, len(observations), 50):
        window = observations[e-window_size:e]
        if last_window is None:
            timeseries = window_to_timeseries([window, "n"])
        else:
            timeseries = update_timeseries(last_window, [window, "n"], window_size, e - last_ob)

        last_window = timeseries
        last_ob = e
        update_timeseries_record.append(timeseries)
    update_end = time.time()

    for standard_timeseries, incre_timeseries in zip(standard_timeseries, update_timeseries_record):
        np.testing.assert_equal(standard_timeseries, incre_timeseries)
    window_size = 250
    standard_timeseries = []
    standard_start = time.time()
    for e in range(250, len(observations), 300):
        window = observations[e-window_size:e]
        timeseries = window_to_timeseries([window, "n"])
        standard_timeseries.append(timeseries)
    standard_end = time.time()

    update_timeseries_record = []
    last_window = None
    last_ob = None
    update_start = time.time()
    for e in range(250, len(observations), 300):
        window = observations[e-window_size:e]
        if last_window is None:
            timeseries = window_to_timeseries([window, "n"])
        else:
            timeseries = update_timeseries(last_window, [window, "n"], window_size, e - last_ob)

        last_window = timeseries
        last_ob = e
        update_timeseries_record.append(timeseries)
    update_end = time.time()

    for standard_timeseries, incre_timeseries in zip(standard_timeseries, update_timeseries_record):
        np.testing.assert_equal(standard_timeseries, incre_timeseries)



def test_window_to_timeseries_update_correct():
    num_features = 10
    observations = []
    for i in range(100):
        features = []
        for f in range(num_features):
            features.append(random.random())
        label = random.randint(0, 2)
        p = random.randint(0, 2)
        e = label == p
        observations.append((features, label, p, e))
    
    window_size = 250
    last_window = None
    last_ob = None
    for e in range(250, len(observations), 1):
        window = observations[e-window_size:e]
        standard_timeseries = window_to_timeseries([window, "n"])
        if last_window is None:
            incre_timeseries = window_to_timeseries([window, "n"])
        else:
            incre_timeseries = update_timeseries(last_window, [window, "n"], window_size, e - last_ob)
        last_window = incre_timeseries
        last_ob = e

        np.testing.assert_equal(standard_timeseries, incre_timeseries)

# test_normalizer_add_stats()
# test_normalizer_ignore_source_add_stats()
# test_normalizer_ignore_feature_add_stats()
# test_normalizer_get_normed_value()
# test_normalizer_get_flat_vec()
# test_normalizer_ignore_source_get_flat_vec()
# test_normalizer_ignore_feature_get_flat_vec()
# test_normalizer_get_flat_labels()
# test_normalizer_get_flat_ranges()
# test_normalizer_norm_vector()
# test_normalizer_order_dict()
# test_window_to_timeseries()
# test_window_to_timeseries_update()
# test_window_to_timeseries_update_correct()
# test_window_to_timeseries_update_correct_no_inplace()

# test_normalizer_get_flat_vec_ignore()
# test_window_to_timeseries_update_time()
# test_window_to_timeseries_update_correct_no_inplace_long_gap()
# test_window_to_timeseries_update_correct_no_inplace_medium_gap()
# test_window_to_timeseries_update_build_start_correct_no_inplace_medium_gap()

# def test_window_to_timeseries_update_time():
#     num_features = 10
#     observations = []
#     for i in range(500):
#         features = []
#         for f in range(num_features):
#             features.append(random.random())
#         label = random.randint(0, 2)
#         p = random.randint(0, 2)
#         e = label == p
#         observations.append((features, label, p, e))
    
#     window_size = 250
#     standard_timeseries = []
#     standard_start = time.time()
#     for e in range(250, len(observations), 1):
#         window = observations[e-window_size:e]
#         timeseries = window_to_timeseries([window, "n"])
#         standard_timeseries.append(timeseries)
#     standard_end = time.time()

#     update_timeseries_record = []
#     last_window = None
#     last_ob = None
#     update_start = time.time()
#     for e in range(250, len(observations), 1):
#         window = observations[e-window_size:e]
#         if last_window is None:
#             timeseries = window_to_timeseries([window, "n"])
#         else:
#             timeseries = update_timeseries(last_window, [window, "n"], window_size, e - last_ob)

#         last_window = timeseries
#         last_ob = e
#         update_timeseries_record.append(timeseries)
#     update_end = time.time()

#     print(standard_end - standard_start)
#     print(update_end - update_start)

# def test_window_to_timeseries_update_correct_no_inplace_long_gap():
#     num_features = 10
#     observations = []
#     for i in range(10000):
#         features = []
#         for f in range(num_features):
#             features.append(random.random())
#         label = random.randint(0, 2)
#         p = random.randint(0, 2)
#         e = label == p
#         observations.append((features, label, p, e))
    
#     window_size = 250
#     standard_timeseries = []
#     standard_start = time.time()
#     for e in range(250, len(observations), 500):
#         window = observations[e-window_size:e]
#         timeseries = window_to_timeseries([window, "n"])
#         standard_timeseries.append(timeseries)
#     standard_end = time.time()

#     update_timeseries_record = []
#     last_window = None
#     last_ob = None
#     update_start = time.time()
#     for e in range(250, len(observations), 500):
#         window = observations[e-window_size:e]
#         if last_window is None:
#             timeseries = window_to_timeseries([window, "n"])
#         else:
#             timeseries = update_timeseries(last_window, [window, "n"], window_size, e - last_ob)

#         last_window = timeseries
#         last_ob = e
#         update_timeseries_record.append(timeseries)
#     update_end = time.time()

#     print(standard_end - standard_start)
#     print(update_end - update_start)
#     for standard_timeseries, incre_timeseries in zip(standard_timeseries, update_timeseries_record):
#         s_features, s_labels, s_ps, s_es, s_eds = standard_timeseries
#         u_features, u_labels, u_ps, u_es, u_eds = incre_timeseries
#         for fi in range(len(s_features)):
#             if not (np.array(s_features[fi]) == np.array(u_features[fi])).all():
#                 print(fi)
#                 print(s_features[fi])
#                 print(u_features[fi])
#                 raise ValueError

#         if not (np.array(s_labels) == np.array(u_labels)).all():
#             print(s_labels)
#             print(u_labels)
#             raise ValueError
#         if not (np.array(s_ps) == np.array(u_ps)).all():
#             print(s_ps)
#             print(u_ps)
#             raise ValueError
#         if not (np.array(s_es) == np.array(u_es)).all():
#             print(s_es)
#             print(u_es)
#             raise ValueError
#         if not (np.array(s_eds) == np.array(u_eds)).all():
#             print(s_eds)
#             print(u_eds)
#             raise ValueError
    
#     print("No inplace change errors for long gap")
# def test_window_to_timeseries_update_correct_no_inplace_medium_gap():
#     num_features = 10
#     observations = []
#     for i in range(10000):
#         features = []
#         for f in range(num_features):
#             features.append(random.random())
#         label = random.randint(0, 2)
#         p = random.randint(0, 2)
#         e = label == p
#         observations.append((features, label, p, e))
    
#     window_size = 250
#     standard_timeseries = []
#     standard_start = time.time()
#     for e in range(250, len(observations), 50):
#         window = observations[e-window_size:e]
#         timeseries = window_to_timeseries([window, "n"])
#         standard_timeseries.append(timeseries)
#     standard_end = time.time()

#     update_timeseries_record = []
#     last_window = None
#     last_ob = None
#     update_start = time.time()
#     for e in range(250, len(observations), 50):
#         window = observations[e-window_size:e]
#         if last_window is None:
#             timeseries = window_to_timeseries([window, "n"])
#         else:
#             timeseries = update_timeseries(last_window, [window, "n"], window_size, e - last_ob)

#         last_window = timeseries
#         last_ob = e
#         update_timeseries_record.append(timeseries)
#     update_end = time.time()

#     print(standard_end - standard_start)
#     print(update_end - update_start)
#     for standard_timeseries, incre_timeseries in zip(standard_timeseries, update_timeseries_record):
#         s_features, s_labels, s_ps, s_es, s_eds = standard_timeseries
#         u_features, u_labels, u_ps, u_es, u_eds = incre_timeseries
#         for fi in range(len(s_features)):
#             if not (np.array(s_features[fi]) == np.array(u_features[fi])).all():
#                 print(fi)
#                 print(s_features[fi])
#                 print(u_features[fi])
#                 raise ValueError

#         if not (np.array(s_labels) == np.array(u_labels)).all():
#             print(s_labels)
#             print(u_labels)
#             raise ValueError
#         if not (np.array(s_ps) == np.array(u_ps)).all():
#             print(s_ps)
#             print(u_ps)
#             raise ValueError
#         if not (np.array(s_es) == np.array(u_es)).all():
#             print(s_es)
#             print(u_es)
#             raise ValueError
#         if not (np.array(s_eds) == np.array(u_eds)).all():
#             print(s_eds)
#             print(u_eds)
#             raise ValueError
    
#     print("No inplace change errors for medium gap")
# def test_window_to_timeseries_update_build_start_correct_no_inplace_medium_gap():
#     num_features = 10
#     observations = []
#     for i in range(10000):
#         features = []
#         for f in range(num_features):
#             features.append(random.random())
#         label = random.randint(0, 2)
#         p = random.randint(0, 2)
#         e = label == p
#         observations.append((features, label, p, e))
    
#     window_size = 250
#     standard_timeseries = []
#     standard_start = time.time()
#     for e in range(50, len(observations), 50):
#         window = observations[max(e-window_size, 0):e]
#         timeseries = window_to_timeseries([window, "n"])
#         standard_timeseries.append(timeseries)
#     standard_end = time.time()

#     update_timeseries_record = []
#     last_window = None
#     last_ob = None
#     update_start = time.time()
#     for e in range(50, len(observations), 50):
#         window = observations[max(e-window_size, 0):e]
#         if last_window is None:
#             timeseries = window_to_timeseries([window, "n"])
#         else:
#             timeseries = update_timeseries(last_window, [window, "n"], window_size, e - last_ob)

#         last_window = timeseries
#         last_ob = e
#         update_timeseries_record.append(timeseries)
#     update_end = time.time()

#     print(standard_end - standard_start)
#     print(update_end - update_start)
#     for standard_timeseries, incre_timeseries in zip(standard_timeseries, update_timeseries_record):
#         s_features, s_labels, s_ps, s_es, s_eds = standard_timeseries
#         u_features, u_labels, u_ps, u_es, u_eds = incre_timeseries
#         for fi in range(len(s_features)):
#             if not (np.array(s_features[fi]) == np.array(u_features[fi])).all():
#                 print(fi)
#                 print(s_features[fi])
#                 print(u_features[fi])
#                 raise ValueError

#         if not (np.array(s_labels) == np.array(u_labels)).all():
#             print(s_labels)
#             print(u_labels)
#             raise ValueError
#         if not (np.array(s_ps) == np.array(u_ps)).all():
#             print(s_ps)
#             print(u_ps)
#             raise ValueError
#         if not (np.array(s_es) == np.array(u_es)).all():
#             print(s_es)
#             print(u_es)
#             raise ValueError
#         if not (np.array(s_eds) == np.array(u_eds)).all():
#             print(s_eds)
#             print(u_eds)
#             raise ValueError
    
#     print("No inplace change errors for medium gap")

# def test_normalizer_get_flat_vec_ignore():
#     normalizer = Normalizer(ignore_sources=["Labels"])
#     normalizer2 = Normalizer(ignore_features=["FI"])
#     stats = {
#         "Feature": {"Mean": 100, "FI": 25},
#         "Labels": {"Mean": 0.5, "FI": 50},
#     }

#     normalizer.add_stats(stats)
#     normalizer2.add_stats(stats)
#     stats_2 = {
#         "Feature": {"Mean": 150, "FI": 25},
#         "Labels": {"Mean": 0.5, "FI": 25},
#     }

#     normalizer.add_stats(stats_2)
#     normalizer2.add_stats(stats_2)
#     stats_3 = {
#         "Feature": {"Mean": 125, "FI": 50},
#         "Labels": {"Mean": 0.5, "FI": 10},
#     }
#     normalizer.add_stats(stats_3)
#     normalizer2.add_stats(stats_3)

#     print(normalizer.get_flat_vector(stats_3, use_full=True))
#     print(normalizer.get_flat_vector(stats_3, use_full=False))
#     print(normalizer2.get_flat_vector(stats_3, use_full=True))
#     print(normalizer2.get_flat_vector(stats_3, use_full=False))