import math
import sys
from copy import deepcopy

import numpy as np
from ConceptFingerprint.Classifier.feature_selection.mutual_information import *
from ConceptFingerprint.Classifier.matrixSketch import FrequentDirections


class Fingerprint:
    def __init__(self, stats, normalizer, performance_sources=['labels', 'errors', 'error_distances', 'predictions'], performance_features=['FI'], num_bins=None):
        self.id = 0
        self.seen = 1
        self.seen_since_performance_reset = 1
        self.sources = set()
        self.features = set()
        self.normalizer = normalizer
        self.fingerprint = {}
        self.fingerprint_values = {}
        self.flat_total_vec = np.empty(self.normalizer.total_num_signals)
        self.flat_ignore_vec = np.empty(self.normalizer.ignore_num_signals)
        self.performance_sources = performance_sources
        self.performance_features = performance_features
        self.dirty_data = False
        self.dirty_performance = False
        for source in stats.keys():
            self.sources.add(source)
            if source not in self.fingerprint:
                self.fingerprint[source] = {}
                self.fingerprint_values[source] = {}
            for feature in stats[source].keys():
                self.features.add(feature)
                value = stats[source][feature]
                if feature not in self.fingerprint[source]:
                    self.fingerprint[source][feature] = self.reset_feature(value)
                    self.fingerprint_values[source][feature] = value

    def reset_feature(self, value, seen=1):
        return {"value": value, "stdev": 0, "seen": seen, "M": value, "S": 0}

    def __str__(self):
        return str(self.fingerprint)

    def __repr__(self):
        return str(self)

    def update_online_stats(self, source, feature, value, weight=1):
        current_value = self.fingerprint[source][feature]["value"]
        current_weight = self.fingerprint[source][feature]["seen"]
        # Take the weighted average of current and new
        new_value = ((current_value * current_weight) +
                     (value * weight)) / (current_weight + weight)

        # Formula for online stdev
        k = self.fingerprint[source][feature]["seen"] + 1
        last_M = self.fingerprint[source][feature]["M"]
        last_S = self.fingerprint[source][feature]["S"]

        new_M = last_M + (value - last_M)/k
        new_S = last_S + (value - last_M)*(value - new_M)

        variance = new_S / (k - 1)
        stdev = math.sqrt(variance) if variance > 0 else 0

        self.fingerprint[source][feature]["value"] = new_value
        self.fingerprint[source][feature]["stdev"] = stdev
        self.fingerprint[source][feature]["seen"] = k
        self.fingerprint[source][feature]["M"] = new_M
        self.fingerprint[source][feature]["S"] = new_S
        self.fingerprint_values[source][feature] = new_value

    def incorperate(self, stats, weight=1):
        self.seen += 1
        self.seen_since_performance_reset += 1
        for source in stats.keys():
            for feature in stats[source].keys():
                flat_total_index = self.normalizer.total_indexes[source][feature]
                value = stats[source][feature]
                self.update_online_stats(source, feature, value, weight)
                self.flat_total_vec[flat_total_index] = self.fingerprint_values[source][feature]
                if source in self.normalizer.ignore_indexes and feature in self.normalizer.ignore_indexes[source]:
                    flat_ignore_index = self.normalizer.ignore_indexes[source][feature]
                    self.flat_ignore_vec[flat_ignore_index] = self.fingerprint_values[source][feature]

    def incorperate_evolution_stats(self, stats, weight=1):
        self.seen += 1
        self.seen_since_performance_reset = 1
        for source in stats.keys():
            for feature in stats[source].keys():
                flat_total_index = self.normalizer.total_indexes[source][feature]
                if not(source in self.performance_sources or feature in self.performance_features):
                    continue
                value = stats[source][feature]
                self.fingerprint[source][feature] = self.reset_feature(value)
                self.fingerprint_values[source][feature] = value
                self.flat_total_vec[flat_total_index] = self.fingerprint_values[source][feature]
                if source in self.normalizer.ignore_indexes and feature in self.normalizer.ignore_indexes[source]:
                    flat_ignore_index = self.normalizer.ignore_indexes[source][feature]
                    self.flat_ignore_vec[flat_ignore_index] = self.fingerprint_values[source][feature]

    def initiate_evolution_plasticity(self):
        """ Reset the seen and standard deviation on performance
        statistics, so we can easily incorperate new information
        from a new evolution in behaviour.
        As old behaviour is not accurate, we lower the weight.
        """
        self.seen_since_performance_reset = 1
        for source in self.sources:
            for feature in self.features:
                flat_total_index = self.normalizer.total_indexes[source][feature]
                if not(source in self.performance_sources or feature in self.performance_features):
                    continue
                value = self.fingerprint_values[source][feature]
                self.fingerprint[source][feature] = self.reset_feature(value, seen=0.1)
                self.flat_total_vec[flat_total_index] = self.fingerprint_values[source][feature]
                if source in self.normalizer.ignore_indexes and feature in self.normalizer.ignore_indexes[source]:
                    flat_ignore_index = self.normalizer.ignore_indexes[source][feature]
                    self.flat_ignore_vec[flat_ignore_index] = self.fingerprint_values[source][feature]

    def initiate_clean_evolution_plasticity(self, clean_starting_stats):
        """ Reset the seen and standard deviation on performance
        statistics using a clean base, so we can easily incorperate new information
        from a new evolution in behaviour.
        """
        self.seen_since_performance_reset = 1
        for source in self.sources:
            for feature in self.features:
                flat_total_index = self.normalizer.total_indexes[source][feature]
                if not(source in self.performance_sources or feature in self.performance_features):
                    continue
                value = clean_starting_stats[source][feature]
                self.fingerprint[source][feature] = self.reset_feature(value)
                self.fingerprint_values[source][feature] = value
                self.flat_total_vec[flat_total_index] = self.fingerprint_values[source][feature]
                if source in self.normalizer.ignore_indexes and feature in self.normalizer.ignore_indexes[source]:
                    flat_ignore_index = self.normalizer.ignore_indexes[source][feature]
                    self.flat_ignore_vec[flat_ignore_index] = self.fingerprint_values[source][feature]

    def merge_performance_data(self, performance_fingerprint):
        self.seen += 1
        self.seen_since_performance_reset = 1
        for source in self.sources:
            for feature in self.features:
                flat_total_index = self.normalizer.total_indexes[source][feature]
                if (source in self.performance_sources or feature in self.performance_features):
                    continue
                self.fingerprint[source][feature] = deepcopy(
                    performance_fingerprint.fingerprint[source][feature])
                self.fingerprint_values[source][feature] = deepcopy(
                    performance_fingerprint.fingerprint_values[source][feature])
                self.flat_total_vec[flat_total_index] = self.fingerprint_values[source][feature]
                if source in self.normalizer.ignore_indexes and feature in self.normalizer.ignore_indexes[source]:
                    flat_ignore_index = self.normalizer.ignore_indexes[source][feature]
                    self.flat_ignore_vec[flat_ignore_index] = self.fingerprint_values[source][feature]

    def get_performance_values(self):
        performance_values = {}
        for source in self.sources:
            for feature in self.features:
                if not(source in self.performance_sources or feature in self.performance_features):
                    continue
                if source not in performance_values:
                    performance_values[source] = {}
                performance_values[source][feature] = self.fingerprint_values[source][feature]
        return performance_values


class FingerprintCache(Fingerprint):
    """Fingerprint extended with a cache for calculating feature histograms
    """
    def __init__(self, stats, normalizer, performance_sources=['labels', 'errors', 'error_distances', 'predictions'], performance_features=['FI'], num_bins=None):
        self.cached_bins = {}
        self.cache_range = {}
        self.cache_dirty = True
        self.num_bins = num_bins
        super().__init__(stats, normalizer, performance_sources, performance_features, num_bins)

    def incorperate(self, stats, weight=1):
        self.cache_dirty = True
        super().incorperate(stats, weight)
        
    def incorperate_evolution_stats(self, stats, weight=1):
        self.cache_dirty = True
        super().incorperate_evolution_stats(stats, weight)
       
    def initiate_evolution_plasticity(self):
        self.cache_dirty = True
        super().initiate_evolution_plasticity()
        
    def initiate_clean_evolution_plasticity(self, clean_starting_stats):
        """ Reset the seen and standard deviation on performance
        statistics using a clean base, so we can easily incorperate new information
        from a new evolution in behaviour.
        """
        self.cache_dirty = True
        super().initiate_clean_evolution_plasticity(clean_starting_stats)
        
    def merge_performance_data(self, performance_fingerprint):
        self.cache_dirty = True
        super().merge_performance_data(performance_fingerprint)

    def reset_dirty_cache(self):
        if self.cache_dirty:
            self.cached_bins = {}
            self.cache_range = {}
            self.cache_dirty = False
    
    def check_cache_size(self, cache, range_tuple, source, feature):
        # We only cache the last 10 values, so if we are storing more than 10, reset and store only current query.
        if len(cache[source][feature]) > 10:
            save_val = cache[source][feature][range_tuple]
            cache[source][feature] = {}
            cache[source][feature][range_tuple] = save_val

    def get_range_if_cached(self, range_tuple, source, feature):
        if source in self.cached_bins and feature in self.cached_bins[source] and range_tuple in self.cached_bins[source][feature]:
            self.check_cache_size(self.cached_bins, range_tuple, source, feature)
            return self.cached_bins[source][feature][range_tuple]
        return None

    def get_binned_feature(self, source, feature, feature_range):
        """Get a histogram of a feature, given a [max, min] range.
        Incorporates caching, if we have generated a histogram previously and not updated the data since, we reuse.
        """
        # If cache is dirty (i.e. stats have been incorperated since last generated)
        # We regenerate. Otherwise, we reuse.
        self.reset_dirty_cache()
        
        # Check if we have computed and stored the histogram for a given range.
        range_tuple = (feature_range[0], feature_range[1])
        cache_val = self.get_range_if_cached(range_tuple, source, feature)
        if cache_val:
            return cache_val
        
        # Init dict caches
        if source not in self.cached_bins:
            self.cached_bins[source] = {}
            self.cache_range[source] = {}
        if feature not in self.cached_bins[source]:
            self.cached_bins[source][feature] = {}

        # Otherwise, recalculate
        feature_data = self.fingerprint[source][feature]
        # Set 0.01 as minimum possible standard deviation to avoid errors when we haven't seen enough
        bin_counts, bin_positions = bin_X(feature_data['value'], max(
            feature_data['stdev'], 0.01), feature_range[0], feature_range[1], self.num_bins, feature_data['seen'])
        

        self.cached_bins[source][feature][range_tuple] = (
            bin_counts, bin_positions)
        self.cache_range[source][feature] = feature_range

        self.check_cache_size(self.cached_bins, range_tuple, source, feature)
        return self.cached_bins[source][feature][range_tuple]


class FingerprintBinningCache(FingerprintCache):
    """ Fingerprint which stores a real histogram of features.
    """
    def __init__(self, stats, normalizer, performance_sources=['labels', 'errors', 'error_distances', 'predictions'], performance_features=['FI'], num_bins=25):
        super().__init__(stats, normalizer,performance_sources,performance_features,num_bins)

    def reset_feature(self, value, seen=1):
        return {"value": value, "stdev": 0, "seen": seen, "M": value, "S": 0, 'Range': [value, value], 'Bins': [(value, value) for i in range(self.num_bins)], 'BinsNP': np.full(self.num_bins + 1, value), 'Histogram': [1, *[0 for i in range(1, self.num_bins)]]}

    def check_bins(self, source, feature, value):
        """
        Check value fits in bins well.
        Simple binning partitions range evenly,
        so value fits if in range.
        """
        mi_range = self.fingerprint[source][feature]['Range']
        in_range = mi_range[0] <= value <= mi_range[1]
        if not in_range:
            self.update_bins(source, feature, value)

    def update_bins(self, source, feature, value):
        """
        We update the bin boundaries.
        Need to shift current bin items into new bins.
        """
        old_range = self.fingerprint[source][feature]['Range']
        new_range = [min(old_range[0], value), max(old_range[1], value)]
        current_bins = self.fingerprint[source][feature]["Bins"]
        current_histogram = self.fingerprint[source][feature]["Histogram"]

        # Extract current items in histogram
        # Note: values have to be estimated
        current_items = {}
        for bin_range, items in zip(current_bins, current_histogram):
            estimated_bin_value = (bin_range[0] + bin_range[1]) / 2
            current_items[estimated_bin_value] = current_items.get(
                estimated_bin_value, 0) + items

        # Create new binning based on new range
        # Note: We create both a numpy style binning
        # and tuple based binning, which both
        # describe the same thing.
        new_bins = []
        new_bins_np = np.empty(self.num_bins + 1)
        total_width = new_range[1] - new_range[0]
        bin_width = total_width / self.num_bins
        for b_i in range(self.num_bins):
            i = new_range[0] + b_i * bin_width
            new_bins.append((i, i+bin_width))
            new_bins_np[b_i] = i
            new_bins_np[b_i + 1] = i + bin_width

        new_histogram = [0 for i in range(self.num_bins)]
        for val, count in current_items.items():
            new_histogram[self.get_bin(val, new_bins)] += count

        self.fingerprint[source][feature]['Range'] = new_range
        self.fingerprint[source][feature]['Bins'] = new_bins
        self.fingerprint[source][feature]['BinsNP'] = new_bins_np
        self.fingerprint[source][feature]['Histogram'] = new_histogram

    def update_histogram(self, source, feature, value):
        self.fingerprint[source][feature]["Histogram"][self.get_bin(
            value, source=source, feature=feature)] += 1

    def get_bin_fast(self, value, bins):
        """ Bin value in O(1) given range.
        Assumes bins equally partition range.
        """
        bins_start = bins[0][0]
        bin_width = bins[0][1] - bins[0][0]
        if bin_width == 0:
            return len(bins) - 1
        dist_from_start = value - bins_start
        bin_i = int(dist_from_start // bin_width)
        bin_i = min(bin_i, len(bins) - 1)
        return bin_i

    def get_bin(self, value, bins=None, source=None, feature=None):
        if not bins:
            bins = self.fingerprint[source][feature]["Bins"]
        return self.get_bin_fast(value, bins)

    def incorperate(self, stats, weight=1):
        super().incorperate(stats, weight)
        for source in stats.keys():
            for feature in stats[source].keys():
                value = stats[source][feature]
                self.check_bins(source, feature, value)
                self.update_histogram(source, feature, value)

    def rebin_histogram(self, histogram, original_bins, new_bins):
        values = {}
        min_val = None
        max_val = None
        num_bins = len(original_bins)

        for count, (bin_start, bin_end) in zip(histogram, original_bins):
            val = (bin_start + bin_end) / 2
            if val not in values:
                values[val] = 0
            values[val] += count

        new_histogram = [0 for i in range(num_bins)]
        for val in values:
            new_histogram[self.get_bin(val, new_bins)] += values[val]
        return new_histogram, new_bins

    def rebin_histogram_np(self, histogram, original_bins, new_bins):
        num_bins = len(original_bins) - 1
        bin_centers = (original_bins[1:] + original_bins[:-1]) / 2.
        hist_values = np.repeat(bin_centers, histogram)

        new_histogram, new_bins = np.histogram(hist_values, bins=new_bins)
        return new_histogram, new_bins

    def get_binned_feature(self, source, feature, overall_bins, np_bins=None):
        # If cache is dirty (i.e. stats have been incorperated since last generated)
        # We regenerate. Otherwise, we reuse.
        self.reset_dirty_cache()

        # Check if range is in cache
        if np_bins is not None:
            feature_range = (np_bins[0], np_bins[-1])
        else:
            feature_range = (overall_bins[0][0], overall_bins[-1][1])
        range_tuple = (feature_range[0], feature_range[1])
        cache_val = self.get_range_if_cached(range_tuple, source, feature)
        if cache_val:
            return cache_val

        if source not in self.cached_bins:
            self.cached_bins[source] = {}
            self.cache_range[source] = {}
        if feature not in self.cached_bins[source]:
            self.cached_bins[source][feature] = {}
        
        feature_data = self.fingerprint[source][feature]
        feature_hist = feature_data["Histogram"]
        feature_bins = feature_data["BinsNP"] if np_bins is not None else feature_data["Bins"]
        
        # Check if range is current binning, and can return directly
        # and add to cache
        current_bin_range = (
            feature_data['Bins'][0][0], feature_data['Bins'][-1][1])
        if (current_bin_range[0], current_bin_range[1]) == (feature_range[0], feature_range[1]):
            self.cached_bins[source][feature][range_tuple] = (feature_hist, feature_bins)
        else:
        # Otherwise, generate the binning for the new range, store it and return
            rebin_hist_func = self.rebin_histogram_np if np_bins is not None else self.rebin_histogram
            new_histogram, new_bins = rebin_hist_func(feature_hist, feature_bins, overall_bins)
            self.cached_bins[source][feature][range_tuple] = (
                new_histogram, new_bins)
        self.check_cache_size(self.cached_bins, range_tuple, source, feature)
        return self.cached_bins[source][feature][range_tuple]


class FingerprintSketchCache(FingerprintCache):
    def __init__(self, stats, normalizer, performance_sources=['labels', 'errors', 'error_distances', 'predictions'], performance_features=['FI'], num_bins=25):
        super().__init__(stats, normalizer, performance_sources, performance_features, num_bins)
        self.cached_covariance = None
        self.sketch = FrequentDirections(
            d=self.normalizer.ignore_num_signals, ell=self.num_bins)
        for source in stats.keys():
            for feature in stats[source].keys():
                value = stats[source][feature]
                self.reset_sketch_feature(source, feature, value)
        self.sketch.nextZeroRow = 1

    def get_feature_range(self, source, feature):
        """ Calculate the range of a feature given a matrix sketch
        """
        self.reset_dirty_cache()
        if source in self.cache_range and feature in self.cache_range[source]:
            return self.cache_range[source][feature]
        flat_ignore_index = self.normalizer.ignore_indexes[source][feature]
        feature_values = self.sketch.get_column_min(flat_ignore_index)
        min_val = np.nanmin(feature_values)
        max_val = np.nanmax(feature_values)
        if np.isnan(feature_values.min()):
            print("feature values is nan!!!")
        feature_range = (min_val, max_val)
        if source not in self.cache_range:
            self.cache_range[source] = {}
        self.cache_range[source][feature] = feature_range
        return self.cache_range[source][feature]

    def reset_sketch_feature(self, source, feature, value):
        """ Reset the value of a feature in a given sketch
        """
        flat_ignore_index = self.normalizer.ignore_indexes[source][feature]
        reset_col = self.sketch.make_reset_column(value)
        self.sketch._sketch[:self.num_bins, flat_ignore_index] = reset_col

    def reset_sketch_perf_features(self, stats):
        for source in stats.keys():
            for feature in stats[source].keys():
                if not(source in self.performance_sources or feature in self.performance_features):
                    continue
                value = stats[source][feature]
                self.reset_sketch_feature(source, feature, value)

    def reset_feature(self, value, seen=1):
        return {"value": value, "stdev": 0, "seen": seen, "M": value, "S": 0, 'Range': [value, value]}

    def update_sketch(self, new_vec):
        self.sketch.append(new_vec)

    def get_bin_fast(self, value, bins):
        """ Bin value in O(1) given range.
        Assumes bins equally partition range.
        """
        bins_start = bins[0][0]
        bins_end = bins[-1][1]
        bin_width = bins[0][1] - bins[0][0]
        if bin_width == 0:
            return len(bins) - 1
        dist_from_start = value - bins_start
        bin_i = int(dist_from_start // bin_width)
        bin_i = min(bin_i, len(bins) - 1)
        return bin_i

    def get_bin(self, value, bins=None, source=None, feature=None):
        if not bins:
            bins = self.fingerprint[source][feature]["Bins"]
        return self.get_bin_fast(value, bins)

    def incorperate(self, stats, weight=1):
        super().incorperate(stats, weight)
        self.update_sketch(self.flat_ignore_vec)

    def incorperate_evolution_stats(self, stats, weight=1):
        super().incorperate_evolution_stats(stats, weight)
        self.reset_sketch_perf_features(stats)

    def initiate_evolution_plasticity(self):
        super().initiate_evolution_plasticity()
        self.reset_sketch_perf_features(self.fingerprint_values)

    def initiate_clean_evolution_plasticity(self, clean_starting_stats):
        super().initiate_clean_evolution_plasticity(clean_starting_stats)
        self.reset_sketch_perf_features(clean_starting_stats)

    def sketch_col_histogram(self, values, bins):
        min_val = sys.maxsize
        max_val = -(sys.maxsize - 1)
        num_bins = len(bins)

        new_histogram = [0 for i in range(num_bins)]
        for val in values:
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
            new_histogram[self.get_bin(val, bins)] += 1
        return new_histogram, (min_val, max_val)

    def sketch_col_histogram_np(self, values, bins):
        try:
            new_histogram, new_bins = np.histogram(values, bins=bins)
        except Exception as e:
            print(e)
            print(bins)
            raise e

        return new_histogram, (bins[0], bins[1])

    def get_binned_feature(self, source, feature, bins, np_bins=None):
        assert np_bins is not None
        self.reset_dirty_cache()

        # Check if range is in cache
        range_tuple = (bins[0], bins[-1]) if np_bins is not None else (bins[0][0], bins[-1][1])
        bin_cache = self.cached_bins
        range_cache = self.cache_range

        if source in bin_cache and feature in bin_cache[source] and range_tuple in bin_cache[source][feature]:
            self.check_cache_size(bin_cache, range_tuple, source, feature)
            return bin_cache[source][feature][range_tuple]

        if source not in bin_cache:
            bin_cache[source] = {}
            range_cache[source] = {}
        if feature not in bin_cache[source]:
            bin_cache[source][feature] = {}

        # Otherwise, generate the binning for the new range, store it and return
        flat_ignore_index = self.normalizer.ignore_indexes[source][feature]
        feature_values = self.sketch.get_column_min(flat_ignore_index)
        min_val = np.nanmin(feature_values)
        max_val = np.nanmax(feature_values)
        feature_hist_func = self.sketch_col_histogram_np if np_bins is not None else self.sketch_col_histogram
        new_histogram, new_range = feature_hist_func(
            feature_values, bins)
        feature_range = (min_val, max_val)
        range_cache[source][feature] = feature_range
        bin_cache[source][feature][range_tuple] = (
            new_histogram, bins, feature_values)
        self.check_cache_size(bin_cache, range_tuple, source, feature)
        return bin_cache[source][feature][range_tuple]

    def get_binned_covariance(self, source, feature):
        self.reset_dirty_cache()

        flat_ignore_index = self.normalizer.ignore_indexes[source][feature]

        if self.cached_covariance is None:
            sketch = self.sketch.get()
            self.cached_covariance = np.cov(sketch, rowvar=False)
        covariance_matrix = self.cached_covariance

        # We don't want to penalize both features with covariance, we need to include
        # only one. So we use the left triangular covariance matrix, to penalize a feature
        # if it has covariance with a variable with lower index. This means the first one
        # does not get the penalty.

        feature_selected_covariance = covariance_matrix[flat_ignore_index,
                                                        :flat_ignore_index]
        feature_selected_variance = np.diag(
            covariance_matrix[:flat_ignore_index, :flat_ignore_index])
        feature_variance = covariance_matrix[flat_ignore_index,
                                             flat_ignore_index]

        np.seterr('ignore')
        feature_selected_correlation = feature_selected_covariance / \
            (np.sqrt(feature_selected_variance * feature_variance))
        feature_selected_correlation = np.nan_to_num(
            feature_selected_correlation, posinf=0.0)
        feature_selected_abs_correlation = np.abs(feature_selected_correlation)

        # If we are looking at the first feature, there is nothing to compare to so 0 correlation.
        # But downstream expects a list
        if len(feature_selected_covariance) == 0:
            return np.zeros(1), np.zeros(1), np.zeros(1)
        return feature_selected_covariance, feature_selected_correlation, feature_selected_abs_correlation

    def get_binned_feature_np(self, source, feature, bins, np_bins):
        return self.get_binned_feature(source, feature, bins, np_bins=np_bins)
