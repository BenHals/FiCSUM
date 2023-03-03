import math
import numpy as np
from numpy import inf


class Normalizer:
    """ Keeps track of data seen in the stream.
    Keeps track of classes (possible y labels),
    the sources of data, and features.

    For each feature for each source, keeps 
    track of the max and minimum value seen.
    """

    def __init__(self, ignore_sources=None, ignore_features=None, fingerprint_constructor=None):
        self.ignore_sources = []
        if ignore_sources is not None:
            self.ignore_sources = ignore_sources
        self.ignore_features = []
        if ignore_features is not None:
            self.ignore_features = ignore_features
        self.classes = []
        self.sources = set()
        self.features = set()
        self.data_ranges = {}
        self.data_stdev = {}
        self.seen_classes = 0
        self.seen_stats = 0
        self.source_order = None
        self.feature_order = None
        self.total_num_signals = None
        self.total_flat_max = None
        self.total_flat_min = None
        self.ignore_num_signals = None
        self.ignore_flat_max = None
        self.ignore_flat_min = None
        self.changed_since_last_weight_calc = False

        # Store the index used to transform
        # a source-feature key into a flat vector
        self.total_indexes = None
        self.ignore_indexes = None
        self.total_source_ranges = None
        self.ignore_source_ranges = None
        self.fingerprint_constructor = fingerprint_constructor
        self.fingerprint = None

    def add_class(self, c):
        """ Add a class to the list of classes
        we have seen so far.
        """

        if c not in self.classes:
            self.classes.append(c)
            self.classes.sort()
        self.seen_classes += 1

    def init_signals(self, stats):
        """ Initialize normalizer with the 
        sources and features which will be used.
        Should be constant for all observations.
        """
        self.total_indexes = {}
        self.ignore_indexes = {}
        self.total_source_ranges = {}
        self.ignore_source_ranges = {}
        for source in stats.keys():
            self.sources.add(source)
            for feature in stats[source].keys():
                self.features.add(feature)
        self.source_order = sorted([x for x in self.sources])
        self.feature_order = sorted([x for x in self.features])
        index = 0
        ignore_index = 0
        for s in self.source_order:
            self.total_indexes[s] = {}
            self.total_source_ranges[s] = [index]
            if s not in self.ignore_sources:
                self.ignore_indexes[s] = {}
                self.ignore_source_ranges[s] = [ignore_index]
            for f in self.feature_order:
                self.total_indexes[s][f] = index
                index += 1
                if (s not in self.ignore_sources) and (f not in self.ignore_features):
                    self.ignore_indexes[s][f] = ignore_index
                    ignore_index += 1
            self.total_source_ranges[s].append(index)
            if s not in self.ignore_sources:
                self.ignore_source_ranges[s].append(ignore_index)
        self.total_num_signals = index
        self.total_flat_max = np.empty(self.total_num_signals)
        self.total_flat_min = np.empty(self.total_num_signals)
        self.ignore_num_signals = ignore_index
        self.ignore_flat_max = np.empty(self.ignore_num_signals)
        self.ignore_flat_min = np.empty(self.ignore_num_signals)

        self.fingerprint = self.fingerprint_constructor(
            stats=stats, normalizer=self)

    def update_stdev(self, value, source, feature, weight=1):
        current_value = self.data_stdev[source][feature]["value"]
        current_weight = self.data_stdev[source][feature]["seen"]
        # Take the weighted average of current and new
        new_value = ((current_value * current_weight) +
                     (value * weight)) / (current_weight + weight)

        # Formula for online stdev
        k = self.data_stdev[source][feature]["seen"] + 1
        last_M = self.data_stdev[source][feature]["M"]
        last_S = self.data_stdev[source][feature]["S"]

        new_M = last_M + (value - last_M)/k
        new_S = last_S + (value - last_M)*(value - new_M)

        variance = new_S / (k - 1)
        stdev = math.sqrt(variance) if variance > 0 else 0

        self.data_stdev[source][feature]["value"] = new_value
        self.data_stdev[source][feature]["var"] = variance if variance > 0 else 0
        self.data_stdev[source][feature]["stdev"] = stdev
        self.data_stdev[source][feature]["seen"] = k
        self.data_stdev[source][feature]["M"] = new_M
        self.data_stdev[source][feature]["S"] = new_S

    def add_stats(self, stats):
        """ Add a set of statistics. Update
        our stored ranges for each statistic.
        """

        self.seen_stats += 1
        if self.source_order is None:
            self.init_signals(stats)
        else:
            self.fingerprint.incorperate(stats)

        for source in stats.keys():
            if source not in self.data_ranges:
                self.data_ranges[source] = {}
                self.data_stdev[source] = {}
            for feature in stats[source].keys():
                value = stats[source][feature]
                if feature not in self.data_ranges[source]:
                    self.data_ranges[source][feature] = [None, None]
                    self.data_stdev[source][feature] = {
                        "value": value, "var": 0, "stdev": 0, "seen": 1, "M": value, "S": 0}
                else:
                    self.update_stdev(value, source, feature)
                value_range = self.data_ranges[source][feature]
                if value_range[0] is None or value < value_range[0]:
                    value_range[0] = value
                    self.changed_since_last_weight_calc = True
                if value_range[1] is None or value > value_range[1]:
                    value_range[1] = value
                    self.changed_since_last_weight_calc = True
                signal_index = self.total_indexes[source][feature]
                self.total_flat_min[signal_index] = value_range[0]
                self.total_flat_max[signal_index] = value_range[1]
                if (source not in self.ignore_sources) and (feature not in self.ignore_features):
                    ignore_signal_index = self.ignore_indexes[source][feature]
                    self.ignore_flat_min[ignore_signal_index] = value_range[0]
                    self.ignore_flat_max[ignore_signal_index] = value_range[1]

    def get_ignore_index(self, source, feature):
        if (source not in self.ignore_sources) and (feature not in self.ignore_features):
            return self.ignore_indexes[source][feature]
        return None

    def merge(self, other):
        """ Merge ranges with another normalizer.
        Set max and min to the max/min of either.
        """

        for source in self.sources:
            if source not in other.sources:
                continue
            for feature in self.features:
                if feature not in other.data_ranges[source]:
                    continue

                my_range = self.data_ranges[source][feature]
                other_range = other.data_ranges[source][feature]

                my_range[0] = min(my_range[0], other_range[0])
                my_range[1] = max(my_range[1], other_range[1])
                signal_index = self.total_indexes[source][feature]
                self.total_flat_min[signal_index] = my_range[0]
                self.total_flat_max[signal_index] = my_range[1]
                if (source not in self.ignore_sources) and (feature not in self.ignore_features):
                    ignore_signal_index = self.ignore_indexes[source][feature]
                    self.ignore_flat_min[ignore_signal_index] = my_range[0]
                    self.ignore_flat_max[ignore_signal_index] = my_range[1]

    def get_normed_value(self, source, feature, value):
        """ Given a value, and what feature it is,
        return the max min scaled value between 0-1. If the seen range is 0,
        just return the value.
        """

        value_range = self.data_ranges[source][feature]
        width = value_range[1] - value_range[0]
        return (value - value_range[0]) / (width) if width > 0 else 0.5

    def norm_flat_vector(self, flat_vector):
        """ Given a flat vector, we return the min-max normed
        value to between 0-1.
        """
        use_ignore = False
        if flat_vector.shape != self.total_flat_min.shape:
            use_ignore = True
            if flat_vector.shape != self.ignore_flat_min.shape:
                raise ValueError(
                    "Passed flat vector and stored ranges do not have the same shape for full or ignore")
        normed_vector = None
        # This operation may cause divide by 0 or underflow if a MI values has not
        # changed at all over a bin. We set to 0, nan or inf and handle at end.
        if not use_ignore:
            np.seterr(all='ignore')
            normed_vector = (flat_vector - self.total_flat_min) / \
                (self.total_flat_max - self.total_flat_min)

        if use_ignore:
            with np.seterr(divide='ignore', invalid='ignore', under='ignore'):
                normed_vector = (flat_vector - self.ignore_flat_min) / \
                    (self.ignore_flat_max - self.ignore_flat_min)
        normed_vector[np.isnan(normed_vector)] = 0.5
        normed_vector[normed_vector == -inf] = 0
        normed_vector[normed_vector == inf] = 1
        return normed_vector

    def get_norm_scaling_factor(self, source, feature):
        """ Return the min range for a feature, as well
        as the range it covers.
        """
        value_range = self.data_ranges[source][feature]
        width = value_range[1] - value_range[0]
        return value_range[0], width

    def order_dict(self, items, normalize=True, options=None, use_full=False):
        vec = []
        labels = []
        ranges = []
        for source in self.source_order:
            if source in self.ignore_sources and not use_full:
                continue
            for feature in self.feature_order:
                if options and options['meta-features'][feature] != 1:
                    continue
                if feature in self.ignore_features and not use_full:
                    continue
                value = items[source][feature]
                if normalize:
                    normed_value = self.get_normed_value(
                        source, feature, value)
                    vec.append(normed_value)
                else:
                    vec.append(value)
                labels.append(f"{source}-{feature}")
                ranges.append(
                    f"{self.data_ranges[source][feature][0]}:{self.data_ranges[source][feature][1]}")
        if len(vec) == 0:
            raise ValueError(
                "No vector created, check at least one meta-feature enabled in options.")
        return np.array(vec), np.array(labels), np.array(ranges)

    def get_flat_data(self, stats, options=None, use_full=False):
        normed_vec, normed_labels, normed_ranges = self.order_dict(
            stats, normalize=True, options=options, use_full=use_full)
        no_normed_vec, no_normed_labels, no_normed_ranges = self.order_dict(
            stats, normalize=False, options=options, use_full=use_full)
        return normed_vec, no_normed_vec

    def get_flat_vector(self, stats, options=None, use_full=False):
        """Takes in a concept_stat dictionary
        i.e. a dict where dict[source][feature] is the value
        of a feature taken from a timeseries source at
        one point in time
        A normalizer containing the sources, features and min-max
        values for each source feature pair.
        A set of sources to not include.

        And outputs a flat vector of all source feature pair values.
        The order is determined by the sets in the normalizer.
        As sets do not build in a set manner, but should read in one once built,
        This is not neccesarily constant across runs, but should be
        constant across one run of the program.

        """
        normed_vec, normed_labels, normed_ranges = self.order_dict(
            stats, normalize=True, options=options, use_full=use_full)
        no_normed_vec, no_normed_labels, no_normed_ranges = self.order_dict(
            stats, normalize=False, options=options, use_full=use_full)
        return normed_vec, no_normed_vec

    def get_flat_vector_labels(self, stats, options=None, use_full=False):
        """Takes in a concept_stat dictionary
        i.e. a dict where dict[source][feature] is the value
        of a feature taken from a timeseries source at
        one point in time
        A normalizer containing the sources, features and min-max
        values for each source feature pair.
        A set of sources to not include.

        And outputs a flat vector of all source feature pair values.
        The order is determined by the sets in the normalizer.
        As sets do not build in a set manner, but should read in one once built,
        This is not neccesarily constant across runs, but should be
        constant across one run of the program.

        """
        no_normed_vec, no_normed_labels, no_normed_ranges = self.order_dict(
            stats, normalize=False, options=options, use_full=use_full)
        return no_normed_labels

    def get_flat_vector_ranges(self, stats, options=None, use_full=False):
        """Takes in a concept_stat dictionary
        i.e. a dict where dict[source][feature] is the value
        of a feature taken from a timeseries source at
        one point in time
        A normalizer containing the sources, features and min-max
        values for each source feature pair.
        A set of sources to not include.

        And outputs a flat vector of all source feature pair values.
        The order is determined by the sets in the normalizer.
        As sets do not build in a set manner, but should read in one once built,
        This is not neccesarily constant across runs, but should be
        constant across one run of the program.

        """
        no_normed_vec, no_normed_labels, no_normed_ranges = self.order_dict(
            stats, normalize=False, options=options, use_full=use_full)
        return no_normed_ranges
