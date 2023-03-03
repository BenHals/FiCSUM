import logging
import math
import warnings
from collections import deque
from copy import deepcopy

import numpy as np
import scipy.stats
from ConceptFingerprint.Classifier.feature_selection.fisher_score import \
    fisher_score
from ConceptFingerprint.Classifier.feature_selection.mutual_information import *
from ConceptFingerprint.Classifier.normalizer import Normalizer
from ConceptFingerprint.Classifier.metafeature_extraction import (window_to_timeseries,
                                                                  update_timeseries,
                                                                  get_concept_stats_from_base,
                                                                  get_concept_stats)
from ConceptFingerprint.Classifier.fingerprint import (FingerprintCache,
                                                       FingerprintBinningCache,
                                                       FingerprintSketchCache)
from ConceptFingerprint.Classifier.feature_selection.online_feature_selection import (
    feature_selection_None,
    feature_selection_original,
    feature_selection_fisher,
    feature_selection_fisher_overall,
    feature_selection_cached_MI,
    feature_selection_histogramMI,
    feature_selection_histogram_covredMI,
    mi_from_cached_fingerprint_bins,
    mi_from_fingerprint_sketch,
    mi_cov_from_fingerprint_sketch,
    mi_from_fingerprint_histogram_cache)
from scipy.spatial.distance import cosine
from scipy.stats.stats import pearsonr
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.utils import check_random_state, get_dimensions


warnings.filterwarnings('ignore')


def get_dimension_weights(fingerprints, state_active_non_active_fingerprints, normalizer, state_id=None, feature_selection_method="default"):
    """ Use feature selection methods to weight
    each meta-information feature.
    """
    feature_selection_func = None

    if feature_selection_method in ["uniform", "None"]:
        feature_selection_func = feature_selection_None
    if feature_selection_method in ["fisher_overall"]:
        feature_selection_func = feature_selection_fisher_overall
    if feature_selection_method == "fisher":
        feature_selection_func = feature_selection_fisher
    if feature_selection_method in ["default", "original"]:
        feature_selection_func = feature_selection_original

    if feature_selection_method in ["gaussian_approx", "CacheMIHy"]:
        def feature_selection_func(f, naf, n, sid): return feature_selection_cached_MI(
            f, mi_from_cached_fingerprint_bins, naf, n, sid)
    if feature_selection_method in ["sketch", "sketch_MI"]:
        def feature_selection_func(f, naf, n, sid): return feature_selection_histogramMI(
            f, mi_from_fingerprint_sketch, naf, n, sid)
    if feature_selection_method in ["sketch_covariance", "sketch_covMI"]:
        def feature_selection_func(f, naf, n, sid): return feature_selection_histogramMI(
            f, mi_cov_from_fingerprint_sketch, naf, n, sid)
    if feature_selection_method in ["sketch_covariance_weighted", "sketch_covMI_weighted"]:
        def feature_selection_func(f, naf, n, sid): return feature_selection_histogramMI(
            f, mi_cov_from_fingerprint_sketch, naf, n, sid, weighted=True)
    if feature_selection_method in ["sketch_covariance_redundancy", "sketch_covredMI"]:
        def feature_selection_func(f, naf, n, sid): return feature_selection_histogram_covredMI(
            f, mi_from_fingerprint_sketch, naf, n, sid)
    if feature_selection_method in ["histogram", "Cachehistogram_MI"]:
        def feature_selection_func(f, naf, n, sid): return feature_selection_histogramMI(
            f, mi_from_fingerprint_histogram_cache, naf, n, sid)
    if feature_selection_func is None:
        raise ValueError(
            f"no falid feature selection method {feature_selection_method}")
    return feature_selection_func(fingerprints, state_active_non_active_fingerprints, normalizer, state_id)


def get_cosine_distance(A, B, weighted, weights):
    """ Get cosine distance between vectors A and B.
    Weight vectors first if weighted is set.
    """
    try:
        if not weighted:
            c = cosine(A, B)
        else:
            c = cosine(A, B, w=weights)

    except:
        c = np.nan
    if np.isnan(c):
        c = 0 if ((not np.any(A)) and (not np.any(B))) else 1
    return c


def get_pearson_distance(A, B, weighted, weights):
    """ Get pearson distance between vectors A and B.
    Weight vectors first if weighted is set.
    """
    try:
        if not weighted:
            p = pearsonr(A, B)
        else:
            p = pearsonr(A*weights, B*weights)
    except:
        p = [np.nan]
    if np.isnan(p[0]):
        A_is_constant = np.all(A == A[0])
        B_is_constant = np.all(B == B[0])
        p = [0] if (A_is_constant and B_is_constant) else [1]
    return 1 - p[0]


def get_histogram_probability(A, fingerprint, weighted, weights):
    """ Get the probability of vector A being drawn from fingerprint.
    Assumes independance between features, so multiplies probabilities.
    Used as a distance measure.
    """
    total_probability = 0
    for s in fingerprint.normalizer.sources:
        for f in fingerprint.normalizer.features:
            vector_index = fingerprint.normalizer.get_ignore_index(s, f)
            if vector_index is None:
                continue
            histogram = fingerprint.fingerprint[s][f]["Histogram"]
            histogram_total = fingerprint.fingerprint[s][f]["seen"]
            bin_index = fingerprint.get_bin(
                value=A[vector_index], source=s, feature=f)
            bin_count = histogram[bin_index]
            probability = bin_count / histogram_total
            likelihood = probability * weights[vector_index]
            if likelihood > 0:
                total_probability += math.log(probability *
                                              weights[vector_index])
    return total_probability


def make_detector(warn=False, s=1e-5):
    """ Create a drift detector. If warn, create a warning detector with higher sensitivity, to trigger prior to the main detector.
    """
    sensitivity = s * 2 if warn else s
    return ADWIN(delta=sensitivity)


def set_fingerprint_bins(make_fingerprint_func, num_bins):
    def make_fingerprint(*args, **kwargs):
        return make_fingerprint_func(*args, num_bins=num_bins, **kwargs)
    return make_fingerprint


def get_fingerprint_constructor(fingerprint_method, num_bins):
    """ Get the appropriate constructor for the passed options.
    """
    if fingerprint_method in ['descriptive', 'cache']:
        return set_fingerprint_bins(FingerprintCache, num_bins)
    if fingerprint_method in ['histogram', 'cachehistogram']:
        return set_fingerprint_bins(FingerprintBinningCache, num_bins)
    if fingerprint_method in ['sketch', 'cachesketch']:
        return set_fingerprint_bins(FingerprintSketchCache, num_bins)
    raise ValueError("Not a valid fingerprint method")


def get_fingerprint_type(fingerprint_method):
    """ Get the appropriate class type for the passed options.
    """
    if fingerprint_method in ['descriptive', 'cache']:
        return FingerprintCache
    if fingerprint_method in ['histogram', 'cachehistogram']:
        return FingerprintBinningCache
    if fingerprint_method in ['sketch', 'cachesketch']:
        return FingerprintSketchCache
    raise ValueError("Not a valid fingerprint method")


class ConceptState:
    """ Represents a data stream concept.
    Maintains current descriptive information, including a classifier, fingerprint, evolution state and recent performance in stationary conditions.
    """

    def __init__(self, id, learner, fingerprint_update_gap, fingerprint_method, fingerprint_bins):
        self.id = id
        self.classifier = learner
        self.seen = 0
        self.fingerprint = None
        self.fingerprint_cache = []
        self.non_active_fingerprints = {}
        self.non_active_fingerprint_buffer = deque()
        self.fingerprint_update_gap = fingerprint_update_gap
        self.last_fingerprint_update = self.fingerprint_update_gap * -1
        self.active_similarity_record = None

        self.similarity_vector_buffer_length = 50
        self.similarity_vector_buffer = deque()

        self.fingerprint_dirty_data = False
        self.fingerprint_dirty_performance = False
        self.num_dirty_performance = None
        self.trigger_dirty_performance_end = False

        self.current_evolution = self.classifier.evolution
        self.fingerprint_method = fingerprint_method
        self.fingerprint_bins = fingerprint_bins
        self.fingerprint_constructor = get_fingerprint_constructor(
            self.fingerprint_method, self.fingerprint_bins)

    def get_sim_observation_probability(self, sim_value, similarity_calc, similarity_min_stdev, similarity_max_stdev):
        """ Return probability of observing the passed similarity, assuming similarity is distributed according to recently seen similarity values.
        """

        # First, recalculate similarity on a record of recently seen observations.
        # Recalculating accounts for changes in normalization etc which may change how similarity behaves.
        if self.active_similarity_record is not None:
            def similarity_calc_func(x, y, z): return similarity_calc(
                x, state=self.id, fingerprint_to_compare=y, flat_nonorm_current_metainfo=z)
            recent_similarity, normal_stdev = self.get_state_recent_similarity(
                similarity_calc_func)
        else:
            # State has no recent data, so we cannot compare to the normal distribution.
            return 0

        recent_stdev = max(normal_stdev, similarity_min_stdev)
        recent_stdev = min(recent_stdev, similarity_max_stdev)

        # Returns probability of observing a sim_value at least as far from the mean.
        p_val = scipy.stats.norm.sf(
            np.abs(sim_value-recent_similarity), loc=0, scale=normal_stdev) * 2

        return p_val

    def observe(self):
        """ Update counters refering to observation count
        """
        self.seen += 1
        self.trigger_dirty_performance_end = False
        if self.num_dirty_performance is not None:
            self.num_dirty_performance -= 1
            if self.num_dirty_performance < 0:
                self.trigger_dirty_performance_end = True
        if (self.seen - self.last_fingerprint_update) >= self.fingerprint_update_gap:
            self.should_update_fingerprint = True

    def incorp_non_active_fingerprint(self, stats, active_state, normalizer):
        if active_state not in self.non_active_fingerprints:
            self.non_active_fingerprints[active_state] = self.fingerprint_constructor(
                stats, normalizer=normalizer)
        else:
            self.non_active_fingerprints[active_state].incorperate(stats, 1)

    def update_non_active_fingerprint(self, stats, active_state, ex, buffer_age, normalizer):
        self.incorp_non_active_fingerprint(
            stats, active_state, normalizer=normalizer)

    def incorp_fingerprint(self, stats, normalizer):
        if self.fingerprint is None:
            self.fingerprint = self.fingerprint_constructor(
                stats, normalizer=normalizer)
        else:
            self.fingerprint.incorperate(stats, 1)

    def update_fingerprint(self, stats, ex, normalizer):
        self.incorp_fingerprint(stats, normalizer)
        self.last_fingerprint_update = self.seen
        self.should_update_fingerprint = False

    def incorp_similarity(self, similarity, similarity_vector, flat_similarity_vec):
        """ Update current statistics in an online manner.
        """
        if self.active_similarity_record is None:
            self.active_similarity_record = {
                "value": similarity, "stdev": 0, "seen": 1, "M": similarity, "S": 0}
        else:
            value = similarity
            current_value = self.active_similarity_record["value"]
            current_weight = self.active_similarity_record["seen"]
            # Take the weighted average of current and new
            new_value = ((current_value * current_weight) +
                         value) / (current_weight + 1)

            # Formula for online stdev
            k = self.active_similarity_record["seen"] + 1
            last_M = self.active_similarity_record["M"]
            last_S = self.active_similarity_record["S"]

            new_M = last_M + (value - last_M)/k
            new_S = last_S + (value - last_M)*(value - new_M)

            variance = new_S / (k - 1)
            stdev = math.sqrt(variance) if variance > 0 else 0

            self.active_similarity_record["value"] = new_value
            self.active_similarity_record["stdev"] = stdev
            self.active_similarity_record["seen"] = k
            self.active_similarity_record["M"] = new_M
            self.active_similarity_record["S"] = new_S
        self.similarity_vector_buffer.append(
            (similarity, similarity_vector, flat_similarity_vec, deepcopy(self.fingerprint)))
        if len(self.similarity_vector_buffer) >= self.similarity_vector_buffer_length:
            old_sim, old_vec, old_flat_vec, old_fingerprint = self.similarity_vector_buffer.popleft()
            self.remove_similarity(old_sim)

    def remove_similarity(self, similarity):
        """ Update current statistics in an online manner.
        """
        current_value = self.active_similarity_record["value"]
        current_weight = self.active_similarity_record["seen"]
        # Take the weighted average of current and new
        new_value = ((current_value * current_weight) -
                     similarity) / (current_weight - 1)

        # Formula for online stdev
        k = self.active_similarity_record["seen"] - 1
        last_M = self.active_similarity_record["M"]
        last_S = self.active_similarity_record["S"]

        new_M = last_M - (similarity - last_M)/k
        new_S = last_S - (similarity - new_M)*(similarity - last_M)

        variance = new_S / (k - 1)
        stdev = math.sqrt(variance) if variance > 0 else 0

        self.active_similarity_record["value"] = new_value
        self.active_similarity_record["stdev"] = stdev
        self.active_similarity_record["seen"] = k
        self.active_similarity_record["M"] = new_M
        self.active_similarity_record["S"] = new_S

    def add_similarity_record(self, similarity, similarity_vector, flat_similarity_vec, ex):
        self.incorp_similarity(
            similarity, similarity_vector, flat_similarity_vec)

    def get_state_recent_similarity(self, similarity_calc_function):
        """ Want to get an idea of the 'normal' similarity
        to this state. We store this, however there is an issue
        when normalization etc changes, the similarity we have
        stored may be different to the similarity we would calculate
        now. So we also store the 100 <sim_vec_buffer_len> most recent
        vectors for which similarity was calculated. We calculate 
        similarity again, and adjust our similarity record based on the difference.
        """
        similarity_record = self.active_similarity_record["value"]
        similarity_stdev_record = self.active_similarity_record["stdev"]
        adjust_ratios = []
        for old_similarity_reading, similarity_vec, flat_similarity_vec, similarity_fingerprint in list(self.similarity_vector_buffer)[::5]:
            if old_similarity_reading == 0:
                continue
            new_similarity_reading = similarity_calc_function(
                similarity_vec, similarity_fingerprint, flat_similarity_vec)
            adjust_ratios.append(new_similarity_reading /
                                 old_similarity_reading)
        if len(adjust_ratios) > 1:
            average_ratio = min(np.mean(adjust_ratios), 1)
        else:
            average_ratio = 1
        return similarity_record * average_ratio, similarity_stdev_record

    def transition_to(self):
        pass

    def transition_from(self):
        pass

    def start_evolution_transition(self, dirty_length):
        """ Set state to an evolution phase. In this phase, we regard fingerprint meta-info as potentially dirty, as we know that statistics regarding classifier performance must have changed.
        """
        self.current_evolution = self.classifier.evolution

        # We add the current fingerprint, which contains the previous
        # bevaviour to the cache so we can use it while we recreate
        # a fingerprint of new behaviour.
        # For the next window_size, our buffered_window will contain
        # elements with the old behaviour and new, so will not give
        # accurate meta-info. We create a temporary fingerprint to capture
        # this, then when window is clean we start a new fingerprint.
        if self.fingerprint is not None:
            cache_fingerprint = deepcopy(self.fingerprint)
            cache_fingerprint.dirty_performance = self.fingerprint_dirty_performance
            cache_fingerprint.dirty_data = self.fingerprint_dirty_data
            self.fingerprint_cache.append(cache_fingerprint)
            self.fingerprint_cache = self.fingerprint_cache[-6:]
            self.fingerprint.initiate_evolution_plasticity()
            self.fingerprint.id += 1

        self.num_dirty_performance = dirty_length
        self.fingerprint_dirty_performance = True

    def end_evolution_transition(self, clean_base_stats):
        """ End state evolution phase. Called when enough observations have passed to regard data as clean.
        """

        # We add the current fingerprint, a transition period,
        # to the cache.
        # Using a starting set of clean performance stats,
        # from a window now fully containing new behaviour,
        # we again reset plasticity so we can learn.
        if self.fingerprint is not None:
            cache_fingerprint = deepcopy(self.fingerprint)
            cache_fingerprint.dirty_performance = self.fingerprint_dirty_performance
            cache_fingerprint.dirty_data = self.fingerprint_dirty_data
            self.fingerprint_cache.append(cache_fingerprint)
            self.fingerprint_cache = self.fingerprint_cache[-5:]
            self.fingerprint.initiate_clean_evolution_plasticity(
                clean_base_stats)
            self.fingerprint.id += 1

        self.num_dirty_performance = None
        self.fingerprint_dirty_performance = False
        self.trigger_dirty_performance_end = False

    def evolve(self, new_stats, dirty_length):
        """ Trigger evolution. We cache the current fingerprint, so we record meta-info before the change.
        This is used for similarity, as we may want to check for similarity to recent evolutions.
        """
        self.current_evolution = self.classifier.evolution
        if self.fingerprint is not None:
            # We don't want to cache dirty fingerprints.
            # if not self.fingerprint_dirty_performance:
            cache_fingerprint = deepcopy(self.fingerprint)
            cache_fingerprint.dirty_performance = self.fingerprint_dirty_performance
            cache_fingerprint.dirty_data = self.fingerprint_dirty_data
            self.fingerprint_cache.append(cache_fingerprint)
            self.fingerprint_cache = self.fingerprint_cache[-5:]
            self.fingerprint.incorperate_evolution_stats(new_stats)
            self.fingerprint.id += 1

        self.num_dirty_performance = dirty_length

        # If dirty length is set, i.e. we know that for
        # the next window we have observations from
        # different model behaviours, we flag the fingerprint
        # as dirty. This is reset when evolve is called again,
        # which is scheduled for dirty_length observations.
        if dirty_length is not None:
            self.fingerprint_dirty_performance = True
        else:
            self.fingerprint_dirty_performance = False
            self.trigger_dirty_performance_end = False

    def __str__(self):
        return f"<State {self.id}>"

    def __repr__(self):
        return self.__str__()


def check_fs_method(feature_selection_method):
    possible_fingerprints = None
    if feature_selection_method in ["uniform", "None"]:
        possible_fingerprints = [
            "cache", "descriptive", "sketch", "cachesketch", "histogram", "cachehistogram"]
    if feature_selection_method in ["fisher_overall"]:
        possible_fingerprints = [
            "cache", "descriptive", "sketch", "cachesketch", "histogram", "cachehistogram"]
    if feature_selection_method == "fisher":
        possible_fingerprints = [
            "cache", "descriptive", "sketch", "cachesketch", "histogram", "cachehistogram"]
    if feature_selection_method in ["default", "original"]:
        possible_fingerprints = [
            "cache", "descriptive", "sketch", "cachesketch", "histogram", "cachehistogram"]

    if feature_selection_method in ["gaussian_approx", "CacheMIHy"]:
        possible_fingerprints = ["cache", "descriptive"]
    if feature_selection_method in ["sketch", "sketch_MI"]:
        possible_fingerprints = ["sketch", "cachesketch"]
    if feature_selection_method in ["sketch_covariance", "sketch_covMI"]:
        possible_fingerprints = ["sketch", "cachesketch"]
    if feature_selection_method in ["sketch_covariance_weighted", "sketch_covMI_weighted"]:
        possible_fingerprints = ["sketch", "cachesketch"]
    if feature_selection_method in ["sketch_covariance_redundancy", "sketch_covredMI"]:
        possible_fingerprints = ["sketch", "cachesketch"]
    if feature_selection_method in ["histogram", "Cachehistogram_MI"]:
        possible_fingerprints = ["histogram", "cachehistogram"]

    return possible_fingerprints


class DSClassifier:
    def __init__(self,
                 suppress=False,
                 learner=None,
                 sensitivity=0.05,
                 poisson=6,
                 normalizer=None,
                 window_size=150,
                 similarity_gap=10,
                 fingerprint_update_gap=10,
                 non_active_fingerprint_update_gap=50,
                 observation_gap=50,
                 sim_measure="metainfo",
                 MI_calc="metainfo",
                 ignore_sources=None,
                 ignore_features=None,
                 similarity_num_stdevs=3,
                 similarity_min_stdev=0.01,
                 similarity_max_stdev=0.1,
                 buffer_ratio=0.25,
                 feature_selection_method="default",
                 fingerprint_method="auto",
                 fingerprint_bins=None):

        if learner is None:
            raise ValueError('Need a learner')

        self.feature_selection_method = feature_selection_method
        possible_fingerprints = check_fs_method(self.feature_selection_method)
        if possible_fingerprints is None:
            raise ValueError(
                "No matching feature selection method, check fsmethod option")
        if fingerprint_method == "auto":
            self.fingerprint_method = possible_fingerprints[0]
        else:
            if fingerprint_method not in possible_fingerprints:
                raise ValueError(
                    "Fingerprint constructor does not match feature selection method, check fsmethod and fingerprintmethod option")
            self.fingerprint_method = fingerprint_method

        self.fingerprint_bins = fingerprint_bins
        self.fingerprint_constructor = get_fingerprint_constructor(
            self.fingerprint_method, self.fingerprint_bins)
        self.fingerprint_type = get_fingerprint_type(self.fingerprint_method)
        # learner is the classifier used by each state.
        # papers use HoeffdingTree from scikit-multiflow
        self.learner = learner

        # sensitivity is the sensitivity of the concept
        # drift detector
        self.sensitivity = sensitivity
        self.base_sensitivity = sensitivity
        self.current_sensitivity = sensitivity

        # suppress debug info
        self.suppress = suppress

        # rand_weights is if a strategy is setting sample
        # weights for training
        self.rand_weights = poisson > 1

        # poisson is the strength of sample weighting
        # based on leverage bagging
        self.poisson = poisson

        self.in_warning = False
        self.last_warning_point = 0
        self.warning_detected = False

        # initialize waiting state. If we don't have enough
        # data to select the next concept, we wait until we do.
        self.waiting_for_concept_data = False

        # init the current number of states
        self.max_state_id = 0

        # init randomness
        self.random_state = None
        self._random_state = check_random_state(self.random_state)

        self.ex = -1
        self.buffered_ex = -1
        self.classes = None
        self._train_weight_seen_by_model = 0

        self.active_state_is_new = True

        # init data which is exposed to evaluators
        self.found_change = False
        self.num_states = 1
        self.active_state = self.max_state_id
        self.states = []

        self.window_size = window_size
        self.min_window_size = 50
        self.similarity_gap = similarity_gap
        self.fingerprint_update_gap = fingerprint_update_gap
        self.non_active_fingerprint_update_gap = non_active_fingerprint_update_gap
        self.non_active_fingerprint_last_observation = self.non_active_fingerprint_update_gap * -1
        self.similarity_last_observation = self.similarity_gap * -1
        self.ignore_sources = ignore_sources
        self.ignore_features = ignore_features
        self.normalizer = normalizer or Normalizer(
            ignore_sources=self.ignore_sources, ignore_features=self.ignore_features, fingerprint_constructor=self.fingerprint_constructor)
        self.fingerprint_window = deque(maxlen=self.window_size)
        self.fingerprint_similarity_detector = make_detector(s=sensitivity)
        self.fingerprint_similarity_detector_warn = make_detector(
            s=self.get_warning_sensitivity(sensitivity))

        self.take_observations = observation_gap > 0
        self.observation_gap = observation_gap
        self.last_observation = 0
        self.monitor_all_state_active_similarity = None
        self.monitor_all_state_buffered_similarity = None
        self.monitor_active_state_active_similarity = None
        self.monitor_active_state_buffered_similarity = None
        self.fingerprint_update_similarity_mean = deque(maxlen=10)
        self.detected_drift = False

        self.sim_measure = sim_measure
        self.MI_calc = MI_calc

        self.similarity_num_stdevs = similarity_num_stdevs
        self.similarity_min_stdev = similarity_min_stdev
        self.similarity_max_stdev = similarity_max_stdev

        # track the last predicted label
        self.last_label = 0

        # set up repository
        self.state_repository = {}
        init_id = self.max_state_id
        self.max_state_id += 1
        init_state = ConceptState(init_id, self.learner(), self.fingerprint_update_gap,
                                  fingerprint_method=fingerprint_method, fingerprint_bins=self.fingerprint_bins)
        self.state_repository[init_id] = init_state
        self.active_state_id = init_id

        self.manual_control = False
        self.force_transition = False
        self.force_transition_only = False
        self.force_learn_fingerprint = False
        self.force_stop_learn_fingerprint = False
        self.force_transition_to = None
        self.force_transition_to = None
        self.force_lock_weights = False
        self.force_locked_weights = None
        self.force_stop_fingerprint_age = None
        self.force_stop_add_to_normalizer = False

        self.buffer = deque()
        self.buffered_window = deque(maxlen=self.window_size)
        self.buffer_ratio = buffer_ratio
        self.observations_since_last_drift = 0
        self.trigger_point = self.window_size + 1
        self.buffered_observations_since_last_drift = self.cancel_trigger_val()
        self.buffered_observations_since_last_drift_attempt = self.cancel_trigger_val()

        self.trigger_transition_check = False
        self.trigger_attempt_transition_check = False
        self.last_trigger_made_shadow = True
        self.last_transition = None

        self.window_MI_cache = {}
        self.timeseries_cache = {}
        self.all_states_buffered_cache = {}
        self.all_states_active_cache = {}
        self.weights_cache = None
        self.fingerprint_changed_since_last_weight_calc = True

        self.buffered_metainfo = None
        self.buffered_normed_flat = None
        self.buffered_nonormed_flat = None
        self.active_metainfo = None
        self.active_normed_flat = None
        self.active_nonormed_flat = None

        self.monitor_feature_selection_weights = []
        self.monitor_concept_probabilities = {}

    def get_warning_sensitivity(self, s):
        return s * 2

    def get_active_state(self):
        return self.state_repository[self.active_state_id]

    def make_state(self):
        new_id = self.max_state_id
        self.max_state_id += 1
        return new_id, ConceptState(new_id, self.learner(), self.fingerprint_update_gap, fingerprint_method=self.fingerprint_method, fingerprint_bins=self.fingerprint_bins)

    def reset(self):
        pass

    def get_temporal_x(self, X):
        return np.concatenate([X], axis=None)

    def predict(self, X):
        """
        Predict using the model of the currently active state.
        """
        temporal_X = self.get_temporal_x(X)

        return self.get_active_state().classifier.predict([temporal_X])

    def partial_fit(self, X, y, classes=None, sample_weight=None, masked=False):
        """
        Fit an array of observations.
        Splits input into individual observations and
        passes to a helper function _partial_fit.
        Randomly weights observations depending on 
        Config.
        """

        if masked:
            return
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError(
                    'Inconsistent number of instances ({}) and weights ({}).'
                    .format(row_cnt, len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._train_weight_seen_by_model += sample_weight[i]
                    self.ex += 1
                    if self.rand_weights and self.poisson >= 1:
                        k = self.poisson
                        sample_weight[i] = k
                    self._partial_fit(X[i], y[i], sample_weight[i], masked)

    def get_imputed_label(self, X, prediction, last_label):
        """ Get a label.
        Imputes when the true label is masked
        """

        return prediction

    def _metainfo_calc(self, window, window_name, buffered, add_to_normalizer=True, feature_base=None, feature_base_flat=None, stored_shap=None):
        """ Calculate meta-information from a given window.
        We use window_name to cache timeseries calculations, so they can be updated with
        less processing. ONLY USE THE SAME NAME IF THE WINDOW IS GOING TO BE THE SAME!
        buffered is used to calculate how many observations to update.
        If true, we use buffered num updates, which is buffer_ratio slower than normal.

        If no stored_shap is passed, we use the shap of the active model. So pass this if not using this
        model as a base.
        """
        if window_name is not None and window_name in self.timeseries_cache:
            last_timeseries, last_update_time, last_buffered_update_time = self.timeseries_cache[
                window_name]
            num_observations_to_update = self.ex - \
                last_update_time if not buffered else self.buffered_ex - last_buffered_update_time
            current_timeseries = update_timeseries(
                last_timeseries, [window, "n"], self.window_size, num_observations_to_update)
            self.timeseries_cache[window_name] = (
                current_timeseries, self.ex, self.buffered_ex)
        else:
            current_timeseries = window_to_timeseries([window, "n"])
            self.timeseries_cache[window_name] = (
                current_timeseries, self.ex, self.buffered_ex)
        if stored_shap is None:
            stored_shap = self.get_active_state().classifier.shap_model
        if feature_base is None:
            current_metainfo, flat_vec = get_concept_stats(current_timeseries,
                                                           self.get_active_state().classifier,
                                                           stored_shap=stored_shap,
                                                           ignore_sources=self.ignore_sources,
                                                           ignore_features=self.ignore_features,
                                                           normalizer=self.normalizer)
        else:
            current_metainfo, flat_vec = get_concept_stats_from_base(current_timeseries,
                                                                     self.get_active_state().classifier,
                                                                     stored_shap=stored_shap,
                                                                     feature_base=feature_base,
                                                                     feature_base_flat=feature_base_flat,
                                                                     ignore_sources=self.ignore_sources,
                                                                     ignore_features=self.ignore_features,
                                                                     normalizer=self.normalizer)
        if add_to_normalizer and not self.force_stop_add_to_normalizer:
            self.normalizer.add_stats(current_metainfo)

        return current_metainfo, flat_vec

    def _accuracy_calc(self, window, add_to_normalizer=True, feature_base=None):
        accuracy = sum([x[3] for x in window]) / len(window)
        current_metainfo = {"Overall": {"Accuracy": accuracy}}
        if add_to_normalizer and not self.force_stop_add_to_normalizer:
            self.normalizer.add_stats(current_metainfo)

        return current_metainfo, np.array([accuracy])

    def _feature_calc(self, window, add_to_normalizer=True, feature_base=None):
        recent_X = window[-1][0]
        recent_y = window[-1][1]
        x_features = {}
        flat_vec = np.empty(len(recent_X))
        for i, x in enumerate(recent_X):
            feature_name = f"f{i}"
            x_features[feature_name] = x
            flat_vec[i] = x
        current_metainfo = {"Features": x_features}

        if add_to_normalizer and not self.force_stop_add_to_normalizer:
            self.normalizer.add_stats(current_metainfo)

        return current_metainfo, flat_vec

    def _cosine_similarity(self, current_metainfo, state_id=None, fingerprint_to_compare=None, flat_norm_current_metainfo=None, flat_nonorm_current_metainfo=None):
        if state_id is None:
            state_id = self.active_state_id
        if not self.force_lock_weights:

            # Can reuse weights if they exist and were updated this observation, or they haven't changed since last time.
            if self.weights_cache and (not self.normalizer.changed_since_last_weight_calc) and (not self.fingerprint_changed_since_last_weight_calc):
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.weights_cache[
                    0]

            else:
                state_non_active_fingerprints = {k: (v.fingerprint, v.non_active_fingerprints)
                                                 for k, v in self.state_repository.items() if v.fingerprint is not None}
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = get_dimension_weights(list([state.fingerprint for state in self.state_repository.values(
                ) if state.fingerprint is not None]), state_non_active_fingerprints,  self.normalizer, feature_selection_method=self.feature_selection_method)
                self.weights_cache = (
                    (weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights), self.ex)
                self.fingerprint_changed_since_last_weight_calc = False
            self.force_locked_weights = (
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights)
        else:
            weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.force_locked_weights

        self.monitor_feature_selection_weights = sorted_feature_weights

        weights_vec = ignore_flat_weight_vector
        normed_weights = (weights_vec) / (np.max(weights_vec))

        if flat_nonorm_current_metainfo is not None:
            stat_vec, stat_nonorm_vec = self.normalizer.norm_flat_vector(
                flat_nonorm_current_metainfo), flat_nonorm_current_metainfo
        else:
            stat_vec, stat_nonorm_vec = self.normalizer.get_flat_vector(
                current_metainfo)

        # Want to check the current fingerprint,
        # and last clean fingerprint
        # and most recent dirty fingerprint (incase there isn't another clean one and recent is new)
        fingerprints_to_check = []
        if fingerprint_to_compare is None:
            fingerprints_to_check.append(
                self.state_repository[state_id].fingerprint)
            for cached_fingerprint in self.state_repository[state_id].fingerprint_cache[::-1]:
                fingerprints_to_check.append(cached_fingerprint)
                if len(fingerprints_to_check) > 5:
                    break
        else:
            fingerprints_to_check = [fingerprint_to_compare]
        similarities = []
        # NOTE: We actually calculate cosine distance, so we return the MINIMUM DISTANCE
        # This is confusing, as you would think if we were really working with similarity it
        # would be maximum!
        # TODO: rename to distance
        for fp in fingerprints_to_check:
            # fingerprint_vec, fingerprint_nonorm_vec = self.normalizer.get_flat_vector(fp.fingerprint_values)
            fingerprint_nonorm_vec = fp.flat_ignore_vec
            fingerprint_vec = self.normalizer.norm_flat_vector(
                fingerprint_nonorm_vec)
            similarity = get_cosine_distance(
                stat_vec, fingerprint_vec, True, normed_weights)
            similarities.append(similarity)
        min_similarity = min(similarities)

        return min_similarity

    def _sketch_cosine_similarity(self, current_metainfo, state_id=None, fingerprint_to_compare=None, flat_norm_current_metainfo=None, flat_nonorm_current_metainfo=None):
        if state_id is None:
            state_id = self.active_state_id
        if not self.force_lock_weights:

            # Can reuse weights if they exist and were updated this observation, or they haven't changed since last time.
            if self.weights_cache and (not self.normalizer.changed_since_last_weight_calc) and (not self.fingerprint_changed_since_last_weight_calc):
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.weights_cache[
                    0]

            else:
                state_non_active_fingerprints = {k: (v.fingerprint, v.non_active_fingerprints)
                                                 for k, v in self.state_repository.items() if v.fingerprint is not None}
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = get_dimension_weights(list([state.fingerprint for state in self.state_repository.values(
                ) if state.fingerprint is not None]), state_non_active_fingerprints,  self.normalizer, feature_selection_method=self.feature_selection_method)
                self.weights_cache = (
                    (weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights), self.ex)
                self.fingerprint_changed_since_last_weight_calc = False
            self.force_locked_weights = (
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights)
        else:
            weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.force_locked_weights

        self.monitor_feature_selection_weights = sorted_feature_weights

        weights_vec = ignore_flat_weight_vector
        normed_weights = (weights_vec) / (np.max(weights_vec))

        if flat_nonorm_current_metainfo is not None:
            stat_vec, stat_nonorm_vec = self.normalizer.norm_flat_vector(
                flat_nonorm_current_metainfo), flat_nonorm_current_metainfo
        else:
            stat_vec, stat_nonorm_vec = self.normalizer.get_flat_vector(
                current_metainfo)

        # Want to check the current fingerprint,
        # and last clean fingerprint
        # and most recent dirty fingerprint (incase there isn't another clean one and recent is new)
        fingerprints_to_check = []
        if fingerprint_to_compare is None:
            fingerprints_to_check.append(
                self.state_repository[state_id].fingerprint)
            for cached_fingerprint in self.state_repository[state_id].fingerprint_cache[::-1]:
                fingerprints_to_check.append(cached_fingerprint)
                if len(fingerprints_to_check) > 5:
                    break
        else:
            fingerprints_to_check = [fingerprint_to_compare]
        similarities = []
        # NOTE: We actually calculate cosine distance, so we return the MINIMUM DISTANCE
        # This is confusing, as you would think if we were really working with similarity it
        # would be maximum!
        # TODO: rename to distance
        for fp in fingerprints_to_check:
            sketch_similarities = []
            fp_sketch = fp.sketch.get_observation_matrix()
            for sketch_row in range(fp_sketch.shape[0]):
                fingerprint_nonorm_vec = fp_sketch[sketch_row, :]
                fingerprint_vec = self.normalizer.norm_flat_vector(
                    fingerprint_nonorm_vec)
                similarity = get_cosine_distance(
                    stat_vec, fingerprint_vec, True, normed_weights)
                sketch_similarities.append(similarity)
            fingerprint_vec, fingerprint_nonorm_vec = self.normalizer.get_flat_vector(
                fp.fingerprint_values)
            fingerprint_vec = self.normalizer.norm_flat_vector(
                fingerprint_nonorm_vec)
            similarity = get_cosine_distance(
                stat_vec, fingerprint_vec, True, normed_weights)
            similarities.append(np.min(sketch_similarities))
            similarities.append(similarity)
        min_similarity = min(similarities)
        return min_similarity

    def _histogram_similarity(self, current_metainfo, state_id=None, fingerprint_to_compare=None, flat_norm_current_metainfo=None, flat_nonorm_current_metainfo=None):
        if state_id is None:
            state_id = self.active_state_id
        if not self.force_lock_weights:

            # Can reuse weights if they exist and were updated this observation, or they haven't changed since last time.
            if self.weights_cache and (not self.normalizer.changed_since_last_weight_calc) and (not self.fingerprint_changed_since_last_weight_calc):
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.weights_cache[
                    0]

            else:
                state_non_active_fingerprints = {k: (v.fingerprint, v.non_active_fingerprints)
                                                 for k, v in self.state_repository.items() if v.fingerprint is not None}
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = get_dimension_weights(list([state.fingerprint for state in self.state_repository.values(
                ) if state.fingerprint is not None]), state_non_active_fingerprints,  self.normalizer, feature_selection_method=self.feature_selection_method)
                self.weights_cache = (
                    (weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights), self.ex)
                self.fingerprint_changed_since_last_weight_calc = False
            self.force_locked_weights = (
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights)
        else:
            weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.force_locked_weights

        self.monitor_feature_selection_weights = sorted_feature_weights

        weights_vec = ignore_flat_weight_vector
        normed_weights = (weights_vec) / (np.max(weights_vec))

        if flat_nonorm_current_metainfo is not None:
            stat_vec, stat_nonorm_vec = self.normalizer.norm_flat_vector(
                flat_nonorm_current_metainfo), flat_nonorm_current_metainfo
        else:
            stat_vec, stat_nonorm_vec = self.normalizer.get_flat_vector(
                current_metainfo)

        # Want to check the current fingerprint,
        # and last clean fingerprint
        # and most recent dirty fingerprint (incase there isn't another clean one and recent is new)
        fingerprints_to_check = []
        if fingerprint_to_compare is None:
            fingerprints_to_check.append(
                self.state_repository[state_id].fingerprint)
            for cached_fingerprint in self.state_repository[state_id].fingerprint_cache[::-1]:
                fingerprints_to_check.append(cached_fingerprint)
                if len(fingerprints_to_check) > 5:
                    break
        else:
            fingerprints_to_check = [fingerprint_to_compare]
        similarities = []
        for fp in fingerprints_to_check:
            similarity = get_histogram_probability(
                stat_vec, fp, True, normed_weights)
            similarities.append(similarity)
        max_similarity = max(similarities)
        return max_similarity

    def _accuracy_similarity(self, current_metainfo, state=None, fingerprint_to_compare=None):
        if state is None:
            state = self.active_state_id
        state_accuracy = current_metainfo["Overall"]["Accuracy"]
        if fingerprint_to_compare is None:
            fingerprint_accuracy = self.state_repository[
                state].fingerprint.fingerprint_values["Overall"]["Accuracy"]
        else:
            fingerprint_accuracy = fingerprint_to_compare.fingerprint_values[
                "Overall"]["Accuracy"]
        similarity = abs(state_accuracy - fingerprint_accuracy)
        return similarity

    def get_similarity_to_active_state(self, current_metainfo, flat_norm_current_metainfo=None, flat_nonorm_current_metainfo=None):
        return self.get_similarity(current_metainfo, state=self.active_state_id, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo)

    def get_similarity(self, current_metainfo, state=None, fingerprint_to_compare=None, flat_norm_current_metainfo=None, flat_nonorm_current_metainfo=None):
        if self.sim_measure == "metainfo":
            return self._cosine_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo)
        if self.sim_measure == "sketch":
            return self._sketch_cosine_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo)
        if self.sim_measure == "histogram":
            return self._histogram_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo)
        if self.sim_measure == "accuracy":
            return self._accuracy_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare)
        raise ValueError("similarity method not set")

    def get_active_metainfo(self, add_to_normalizer=True, feature_base=None, feature_base_flat=None):
        return self.get_metainfo_from_window(window=self.fingerprint_window, window_name="fingerprint_window", buffered=False, add_to_normalizer=add_to_normalizer, feature_base=feature_base, feature_base_flat=feature_base_flat)

    def get_buffered_metainfo(self, add_to_normalizer=True, feature_base=None, feature_base_flat=None):
        return self.get_metainfo_from_window(window=self.buffered_window, window_name="buffered_window", buffered=True, add_to_normalizer=add_to_normalizer, feature_base=feature_base, feature_base_flat=feature_base_flat)

    def get_metainfo_from_window(self, window, window_name, buffered, add_to_normalizer=True, feature_base=None, feature_base_flat=None, stored_shap=None):
        if self.MI_calc == 'metainfo':
            if self.sim_measure == "metainfo":
                return self._metainfo_calc(window, window_name, buffered, add_to_normalizer=add_to_normalizer, feature_base=feature_base, feature_base_flat=feature_base_flat, stored_shap=stored_shap)
            if self.sim_measure == "sketch":
                return self._metainfo_calc(window, window_name, buffered, add_to_normalizer=add_to_normalizer, feature_base=feature_base, feature_base_flat=feature_base_flat, stored_shap=stored_shap)
            if self.sim_measure == "histogram":
                return self._metainfo_calc(window, window_name, buffered, add_to_normalizer=add_to_normalizer, feature_base=feature_base, feature_base_flat=feature_base_flat, stored_shap=stored_shap)
            if self.sim_measure == "accuracy":
                return self._accuracy_calc(window, add_to_normalizer=add_to_normalizer, feature_base=feature_base)
        if self.MI_calc == "feature":
            return self._feature_calc(window, add_to_normalizer=add_to_normalizer, feature_base=feature_base)
        raise ValueError("MI Method not set correctly")

    def add_to_window_cache(self):
        """ Since the buffered window trails the active window,
        we might be able to reuse calculated meta-info for windows
        which appear as an active window then a buffered window.
        We identify windows by the time we saw the observations at the start
        and end, and cache the calculated meta-info. This should be called right after
        info is calculated.
        We can delete items from the cache when the starting element is less than the start
        of the buffered window. This monotonically increases, so such a window 
        will never match.
        """

        # We don't want to cache when the active window
        # is dirty after an evolution. Since we reclassify
        # buffer, the MI will be different in the buffered_window
        # than the active window.
        if self.get_active_state().fingerprint_dirty_performance:
            return
        active_window_start_time = self.fingerprint_window[0][5]
        active_window_end_time = self.fingerprint_window[-1][5]
        self.window_MI_cache[(active_window_start_time, active_window_end_time)] = (
            self.active_metainfo, self.active_nonormed_flat)
        if len(self.buffered_window) > 0:
            buffered_window_start_time = self.buffered_window[0][5]
            buffered_window_end_time = self.buffered_window[-1][5]
            delete_keys = []
            for cached_range in self.window_MI_cache:
                if cached_range[0] < buffered_window_start_time:
                    delete_keys.append(cached_range)
            for del_key in delete_keys:
                del self.window_MI_cache[del_key]

    def check_window_cache(self, add_to_normalizer=True):
        if len(self.buffered_window) < 1:
            return None, None
        buffered_window_start_time = self.buffered_window[0][5]
        buffered_window_end_time = self.buffered_window[-1][5]
        for cached_range in self.window_MI_cache:
            if cached_range[0] > buffered_window_start_time:
                break
            if cached_range[0] == buffered_window_start_time and cached_range[1] == buffered_window_end_time:
                reused_MI, reused_flat = self.window_MI_cache[cached_range]
                if add_to_normalizer:
                    self.normalizer.add_stats(reused_MI)
                return reused_MI, reused_flat
        return None, None

    def update_and_check_buffer(self, current_observation):
        """ Add an element to the buffer, and roll-over into the training buffered window.
        Pop and buffered window elements and return as new 'clean' data.
        """

        # buffer holds the buffer_size most recent observations.
        # This size is dynamic, and grows as a ratio of observations
        # in the current stationary segment.
        self.buffer.append(current_observation)

        # the buffered observation is assumed to come from the current
        # concept, and be 'safe' from concept drift.
        # I.E we assume we detect any drift during the buffer period.
        # Things can go wrong when this assumption is broken, i.e.
        # we don't detect drift in time.
        # We don't always get a buffered_observation, as the buffer
        # dynamically grows some observations only extend the buffer.
        buffered_observation = None
        self.buffer_length = math.floor(
            self.observations_since_last_drift * self.buffer_ratio)
        if len(self.buffer) >= self.buffer_length:
            buffered_observation = self.buffer.popleft()
            self.buffered_window.append(buffered_observation)

        # The buffered_window holds window_size observations
        # which have been buffered, i.e. should represent the
        # current concept.

        buffered_values = None
        if buffered_observation is not None:
            buffered_values = {
                "X": buffered_observation[0],
                "label": buffered_observation[1],
                "prediction": buffered_observation[2],
                "correctly_classifies": buffered_observation[3],
            }
            self.buffered_ex += 1
            self.buffered_observations_since_last_drift += 1
            self.buffered_observations_since_last_drift_attempt += 1
        return buffered_values, buffered_observation

    def attempt_to_fit(self, buffered_values, sample_weight):
        fit = True
        fit = not self.force_stop_learn_fingerprint
        # Fit the classifier of the current state.
        # We fit on buffered values, to avoid
        # fitting on items from a dufferent concept

        fit = fit and buffered_values is not None
        if fit:
            self.get_active_state().classifier.partial_fit(
                np.asarray([buffered_values["X"]]),
                np.asarray([buffered_values["label"]]),
                sample_weight=np.asarray([sample_weight]),
                classes=self.classes
            )

    def check_evolution(self):
        """ Check if the current state has evolved. 
        If it has, refresh our performance data on buffered information.

        """
        classifier_evolved = self.get_active_state(
        ).classifier.evolution > self.get_active_state().current_evolution

        # If we evolved, we reclassify items in our buffer
        # This is purely for training purposes, as we have
        # already emitted our prediction for these values.
        # These have not been used to fit yet, so this is not biased.
        # This captures the current behaviour of the model.
        # We evolve our state, or reset plasticity of performance
        # features so they can be molded to the new behaviour.
        if classifier_evolved:
            new_buffer = deque()
            for b_X, b_y, b_p, b_cc, b_ev, b_ex in self.buffer:
                new_p = self.get_active_state().classifier.predict([b_X])
                new_cc = b_y == new_p[0]
                new_buffer.append(
                    (b_X, b_y, new_p[0], new_cc, self.get_active_state().classifier.evolution, b_ex))
            self.buffer = new_buffer
            self.window_MI_cache = {}
            self.get_active_state().start_evolution_transition(self.trigger_point)

            if self.get_active_state().fingerprint is not None:
                self.fingerprint_changed_since_last_weight_calc = True

    def monitor_evolution_status(self, buffered_observation):
        """ Should be called at every step, in order to monitor when an evolution should finish.
        """
        if buffered_observation is not None:
            self.get_active_state().observe()

        # When we end the evolution trainsition state, which occurs
        # after window_size observations have passed through the
        # buffer so we have entirely observations of new behaviour,
        # we start a new, clean concept.
        if self.get_active_state().trigger_dirty_performance_end:
            evolved_model_stats = self.get_buffered_metainfo()[0]
            self.get_active_state().end_evolution_transition(evolved_model_stats)
            if self.get_active_state().fingerprint is not None:
                self.fingerprint_changed_since_last_weight_calc = True

    def check_take_measurement(self):
        """
        Check if we need to take a measurement for drift detection.
        This uses the active window, as we are testing if the head of the stream is
        different to the current fingerprint. This should fire relatively often,
        so as to trigger drift detection warnings with minimal delay.
        """
        take_measurement = (
            self.ex - self.similarity_last_observation) >= self.similarity_gap
        take_measurement = take_measurement and len(
            self.buffered_window) >= self.min_window_size
        if self.manual_control:
            take_measurement = self.force_learn_fingerprint and not self.force_stop_learn_fingerprint
        return take_measurement

    def reset_partial_fit_settings(self):
        """ Set class level handlers for the current partial fit
        """
        # When we make a transition, we may have data from the past concept
        # in our window. We wait until this would be entirely flushed, window_size,
        # and try again.
        self.trigger_transition_check = False
        self.trigger_attempt_transition_check = False
        if self.buffered_observations_since_last_drift == self.trigger_point:
            self.trigger_transition_check = True
        if self.buffered_observations_since_last_drift_attempt == self.trigger_point:
            self.trigger_attempt_transition_check = True

        # Global holders of window metainfo, so we can avoid recalculations.
        self.buffered_metainfo = None
        self.buffered_normed_flat = None
        self.buffered_nonormed_flat = None
        self.active_metainfo = None
        self.active_normed_flat = None
        self.active_nonormed_flat = None

        # For # logging
        self.monitor_active_state_active_similarity = None
        self.monitor_active_state_buffered_similarity = None
        self.monitor_all_state_buffered_similarity = None
        self.monitor_all_state_active_similarity = None

    def update_similarity_change_detection(self, similarity):
        """ Update change detectors with active state active similarity deviation.
        """
        # We add similarity to detections directly.
        # This might not be the best method, as it is directional
        # and is not independed, very autocorrelated if this updates more
        # than window_size.
        # Using standard deviations away from normal might help directionality,
        # but not auto correlation.
        if self.get_active_state().active_similarity_record is not None:
            recent_similarity = self.get_active_state(
            ).active_similarity_record["value"]
            normal_stdev = self.get_active_state(
            ).active_similarity_record["stdev"]

            recent_stdev = max(normal_stdev, self.similarity_min_stdev)
            recent_stdev = min(recent_stdev, self.similarity_max_stdev)

            standard_deviations_from_normal = 1 + \
                abs(similarity - recent_similarity) / recent_stdev
            self.fingerprint_similarity_detector.add_element(
                standard_deviations_from_normal)
            self.fingerprint_similarity_detector_warn.add_element(
                standard_deviations_from_normal)

    def check_do_update_fingerprint(self):
        """ Check if active fingerprint should be updated
        """
        # Update fingerprint one state has not been updated for x observations
        # and we have stored enough data to build a window.
        do_update_fingerprint = self.get_active_state().should_update_fingerprint
        do_update_fingerprint = do_update_fingerprint and len(
            self.buffered_window) >= self.min_window_size

        # Extra triggers for control from calling program.
        if self.manual_control:
            do_update_fingerprint = self.force_learn_fingerprint
        do_update_fingerprint = do_update_fingerprint and not self.force_stop_learn_fingerprint
        no_fingerprint_age_limit = self.force_stop_fingerprint_age is None or (
            not (self.get_active_state().seen > self.force_stop_fingerprint_age))
        do_update_fingerprint = do_update_fingerprint and no_fingerprint_age_limit
        return do_update_fingerprint

    def record_active_similarity(self):
        """ Calculate active state similarity to buffered window, and record it.
        Should only be called if self.buffered_meta_info and self.buffered_nonormed_flat are current.
        Requires a valid fingerprint to be active, i.e., one which has seen enough data.
        """
        if is_valid_fingerprint(self.get_active_state().fingerprint):
            similarity = self.get_similarity_to_active_state(
                self.buffered_metainfo, flat_nonorm_current_metainfo=self.buffered_nonormed_flat)
            self.get_active_state().add_similarity_record(similarity, deepcopy(
                self.buffered_metainfo), self.buffered_nonormed_flat, self.ex)
            self.monitor_active_state_buffered_similarity = similarity

    def check_do_update_non_active_fingerprint(self):
        """ Returns if non active fingerprints should be updated.
        """
        # Update non-active fingerprints every x observations
        # and we have stored enough data to build a window.
        do_update_non_active_fingerprint = self.ex - \
            self.non_active_fingerprint_last_observation >= self.non_active_fingerprint_update_gap
        do_update_non_active_fingerprint = do_update_non_active_fingerprint and len(
            self.buffered_window) >= self.min_window_size

        # Extra triggers for control from calling program.
        if self.manual_control:
            do_update_non_active_fingerprint = self.force_learn_fingerprint
        do_update_non_active_fingerprint = do_update_non_active_fingerprint and not self.force_stop_learn_fingerprint
        no_fingerprint_age_limit = self.force_stop_fingerprint_age is None or (
            not (self.get_active_state().seen > self.force_stop_fingerprint_age))
        do_update_non_active_fingerprint = do_update_non_active_fingerprint and no_fingerprint_age_limit

        return do_update_non_active_fingerprint

    def set_current_buffer_metainfo(self):
        """ Extract current meta-info for buffered window from cache and set, or calculate and set.
        Note: The buffered window should already be updated for current data.
        """
        if self.buffered_metainfo is None:
            self.buffered_metainfo, self.buffered_nonormed_flat = self.check_window_cache()
            if self.buffered_metainfo is None:
                self.buffered_metainfo, self.buffered_nonormed_flat = self.get_buffered_metainfo()

    def set_current_active_metainfo(self, add_to_normalizer):
        """ Ensure current active metainfo is set, otherwise calculate and set.
        """
        if self.active_metainfo is None:
            self.active_metainfo, self.active_nonormed_flat = self.get_active_metainfo(
                add_to_normalizer=add_to_normalizer)
            self.add_to_window_cache()

    def make_window_predictions(self, concept_id, cache, current_timestep, observation_window):
        """ Given the ID of a concept, make new predictions for a given window.
        Return the classifier, windows with predictions and list of updates.
        """
        concept_window = []
        concept_classifier = self.state_repository[concept_id].classifier
        if concept_id in cache:
            cached_predictions, cached_errors, cached_timestamp = cache[
                concept_id]
            num_timesteps_to_update = current_timestep - cached_timestamp
            num_elements_to_remove = max(0, len(
                cached_predictions) + num_timesteps_to_update - self.window_size)
            updated_predictions = cached_predictions[
                num_elements_to_remove:]
            updated_errors = cached_errors[num_elements_to_remove:]
        else:
            updated_predictions = []
            updated_errors = []
        j = 0
        for X, y, p, e, ev, ex_i in observation_window:
            if j < len(updated_predictions):
                p = updated_predictions[j]
                e = updated_errors[j]
                concept_window.append((X, y, p, e))
            else:
                p = concept_classifier.predict(X.reshape(1, -1))
                updated_predictions.append(p[0])
                e = y == p[0]
                updated_errors.append(e)
                concept_window.append((X, y, p[0], e))
            j += 1
        return concept_classifier, concept_window, updated_predictions, updated_errors

    def evaluate_all_fingerprint_similarity(self, metainfo, metainfo_flat, observation_window, all_state_cache, current_timestep, use_buffer, add_to_normalizer, update_na_record, update_monitor):

        # Debug data, for printing graph
        observation_accuracy = {}
        observation_stats = {}
        observation_similarities = {}

        # Save model-agnostic features so don't need to recalculate
        fingerprint_feature_base = metainfo

        metainfo_accuracy = sum(
            [w[3] for w in observation_window]) / len(observation_window)

        # We already know info for the active state, so can just reuse
        observation_accuracy["active"] = metainfo_accuracy
        observation_stats["active"] = metainfo

        # Get the stats generated by each stored model
        # To do this, we need to reclassify using that model
        # to get correct meta-information such as error rate.
        # However we can reuse the feature data.
        for concept_id in self.state_repository:
            if self.state_repository[concept_id].fingerprint is None:
                continue
            if concept_id != self.active_state_id:
                concept_classifier, concept_window, updated_predictions, updated_errors = self.make_window_predictions(
                    concept_id=concept_id, cache=all_state_cache, current_timestep=current_timestep, observation_window=observation_window)
                all_state_cache[concept_id] = (
                    updated_predictions, updated_errors, current_timestep)

                window_name = f"buffered_concept_{concept_id}" if use_buffer else f"active_concept_{concept_id}"
                concept_model_stats, concept_model_flat = self.get_metainfo_from_window(window=concept_window,
                                                                                        window_name=window_name,
                                                                                        buffered=use_buffer,
                                                                                        add_to_normalizer=add_to_normalizer,
                                                                                        feature_base=fingerprint_feature_base,
                                                                                        feature_base_flat=metainfo_flat,
                                                                                        stored_shap=concept_classifier.shap_model)
                concept_model_accuracy = sum(
                    [w[3] for w in concept_window]) / len(concept_window)

                if update_na_record:
                    self.state_repository[concept_id].update_non_active_fingerprint(
                        concept_model_stats, self.active_state_id, self.ex, self.buffer_length, normalizer=self.normalizer)
            else:
                concept_model_accuracy = metainfo_accuracy
                concept_model_stats = fingerprint_feature_base
                concept_model_flat = metainfo_flat

            observation_accuracy[concept_id] = concept_model_accuracy
            observation_stats[concept_id] = concept_model_stats
            similarity = self.get_similarity(
                concept_model_stats, state=concept_id, flat_nonorm_current_metainfo=concept_model_flat)
            if update_monitor:
                self.monitor_concept_probabilities[concept_id] = self.state_repository[concept_id].get_sim_observation_probability(
                    similarity, self.get_similarity, self.similarity_min_stdev, self.similarity_max_stdev)

            observation_similarities[concept_id] = similarity
            if concept_id == self.active_state_id:
                observation_similarities["active"] = similarity

        if use_buffer:
            self.monitor_all_state_buffered_similarity = [
                observation_accuracy, observation_stats, self.buffered_window, observation_similarities]
            self.non_active_fingerprint_last_observation = self.ex
            self.fingerprint_changed_since_last_weight_calc = True
        else:
            self.monitor_all_state_active_similarity = [
                observation_accuracy, observation_stats, {}, self.fingerprint_window, observation_similarities]
            self.last_observation = self.ex

    def update_all_fingerprints_buffered(self):
        """ Calculate similarity between ALL states (even non active) on the buffered window.
        This is required for calculating feature weights, so we know which features differ between concepts.
        """

        self.evaluate_all_fingerprint_similarity(metainfo=self.buffered_metainfo, metainfo_flat=self.buffered_nonormed_flat,
                                                 observation_window=self.buffered_window, all_state_cache=self.all_states_buffered_cache,
                                                 current_timestep=self.buffered_ex, use_buffer=True, add_to_normalizer=True, update_na_record=True, update_monitor=False)

    def check_warning_detector(self):
        """ Check if the warning detector has fired. Updates relevant statistics and triggers.
        If we have fired, recreates the detector in order to reset detection from the current position.
        """
        # If the warning detector fires we record the position
        # and reset. We take the most recent warning as the
        # start of out window, assuming this warning period
        # contains elements of the new state.
        if self.fingerprint_similarity_detector_warn.detected_change():
            self.warning_detected = False
            self.in_warning = True
            self.last_warning_point = self.ex
            self.fingerprint_similarity_detector_warn = make_detector(
                s=self.get_warning_sensitivity(self.get_current_sensitivity()))
        if not self.in_warning:
            self.last_warning_point = max(0, self.ex - 100)

    def check_found_change(self, detected_drift):
        """ Checks if we have found change. This either comes from the drift detector triggering, or if we are propagating a past detection forward.
        IMPORTANT: We DONT trigger on good change! If the similarity has increased!
        Propagation occurs if there was not enough data to make a decision at the previous step.
        We also check for manual triggering, for experimentation purposes (evaluating perfect drift.)
        """
        found_change = detected_drift or self.waiting_for_concept_data

        # Just for monitoring
        self.detected_drift = detected_drift or self.force_transition

        # Don't trigger on good changes
        # Or if we haven't tried this model enough to make a fingerprint
        if found_change:
            if self.get_active_state().fingerprint is None:
                found_change = False
            else:
                self.set_current_active_metainfo(add_to_normalizer=True)
                similarity = self.get_similarity_to_active_state(
                    self.active_metainfo)

                def similarity_calc_func(x, y, z): return self.get_similarity(
                    x, state=self.active_state_id, fingerprint_to_compare=y, flat_nonorm_current_metainfo=z)
                normal_similarity, normal_stdev = self.get_active_state(
                ).get_state_recent_similarity(similarity_calc_func)

                # NOTE: We currently use similarity as 0 for most similar and 1 for least.
                # So a better similarity is smaller, worse is larger. So we trigger drift on a larger similarity.
                # This is CONFUSING, so should be updated.
                found_change = found_change and (
                    similarity > normal_similarity)

        if self.manual_control or self.force_transition_only:
            found_change = self.force_transition
            self.trigger_transition_check = False
            self.trigger_attempt_transition_check = False

        return found_change

    def check_take_active_observation(self, found_change):
        take_active_observation = self.take_observations
        take_active_observation = take_active_observation and len(
            self.fingerprint_window) >= self.min_window_size and self.get_active_state().fingerprint is not None
        take_active_observation = take_active_observation and (
            self.ex - self.last_observation >= self.observation_gap or found_change or self.trigger_transition_check)

        return take_active_observation

    def monitor_all_fingerprints_active(self):
        """ Calculate the similarity of ALL states to the current fingerprint window (the most recent data).
        This is just for monitoring purposes at the moment.
        """
        self.set_current_active_metainfo(add_to_normalizer=False)
        self.evaluate_all_fingerprint_similarity(metainfo=self.active_metainfo, metainfo_flat=self.active_nonormed_flat,
                                                 observation_window=self.fingerprint_window, all_state_cache=self.all_states_active_cache,
                                                 current_timestep=self.ex, use_buffer=False, add_to_normalizer=False, update_na_record=False, update_monitor=True)

    def attempt_transition(self):
        """ Attempt to transition states. Evaluates performance, and selects the best performing state to transition too.
        Can transition to the current state.
        Fails if not enough data to calculate best state, in which case the transition attempt is propagated to the next observation.
        """

        self.in_warning = False
        # Find the inactive models most suitable for the current stream. Also return a shadow model
        # trained on the warning period.
        # If none of these have high accuracy, hold off the adaptation until we have the data.
        ranked_alternatives, use_shadow, shadow_model, can_find_good_model = self.rank_inactive_models_for_suitability_fingerprint()

        # Handle manual transitions, for experimentation.
        if self.manual_control and self.force_transition_to is not None:
            if self.force_transition_to in self.state_repository:
                ranked_alternatives = [self.force_transition_to]
                use_shadow = False
                shadow_model = None
                can_find_good_model = True
            else:
                ranked_alternatives = []
                use_shadow = True
                shadow_state = ConceptState(self.force_transition_to, self.learner(
                ), self.fingerprint_update_gap, fingerprint_method=self.fingerprint_method, fingerprint_bins=self.fingerprint_bins)
                shadow_model = shadow_state.classifier
                can_find_good_model = True

        # can_find_good_model is True if at least one model has a close similarity to normal performance.
        if not can_find_good_model:
            # If we did not have enough data to find any good concepts to
            # transition to, wait until we do.
            self.waiting_for_concept_data = True
            return

        # We don't want to replace a shadow model with another
        # shadow on the associated evaluation trigger.
        if (self.trigger_transition_check or self.trigger_attempt_transition_check) and self.last_trigger_made_shadow and use_shadow:
            use_shadow = False
            ranked_alternatives = [self.active_state_id]
        self.last_trigger_made_shadow = use_shadow

        # we need handle a proper transition if this is a transition to a different model.
        proper_transition = use_shadow or (
            ranked_alternatives[-1] != self.active_state_id)
        if proper_transition:
            self.get_active_state().transition_from()

        # If we determined the shadow is the best model, we mark it as a new state,
        # copy the model trained on the warning period across and set as the active state
        if use_shadow:
            self.active_state_is_new = True
            shadow_id, shadow_state = self.make_state()
            shadow_state.classifier = shadow_model
            self.state_repository[shadow_id] = shadow_state
            self.active_state_id = shadow_id
        else:
            # Otherwise we just set the found state to be active
            self.active_state_is_new = False
            transition_target_id = ranked_alternatives[-1]
            self.active_state_id = transition_target_id

        if proper_transition:
            # We reset drift detection as the new performance will not
            # be the same, and reset history for the new state.
            current_sensitivity = self.get_current_sensitivity()
            self.get_active_state().transition_to()
            self.waiting_for_concept_data = False
            self.fingerprint_similarity_detector = make_detector(
                s=current_sensitivity)
            self.fingerprint_similarity_detector_warn = make_detector(
                s=self.get_warning_sensitivity(current_sensitivity))
            self.fingerprint_update_similarity_mean = deque(maxlen=10)
            self.buffer = deque()
            self.buffered_window = deque(maxlen=self.window_size)
            self.fingerprint_window = deque(maxlen=self.window_size)
            self.timeseries_cache = {}

            # Set triggers to future evaluation of this attempt
            self.observations_since_last_drift = self.reset_trigger_val()
            self.buffered_observations_since_last_drift = self.reset_trigger_val()

            # Cancel propagating drift attempt trigger
            self.buffered_observations_since_last_drift_attempt = self.cancel_trigger_val()

        else:
            # Only set trigger on legit detection, or we will cause repeated checks.
            if not (self.trigger_transition_check or self.trigger_attempt_transition_check):
                self.buffered_observations_since_last_drift_attempt = self.reset_trigger_val()

        # If the current attempt was due to a trigger, cancel the trigger
        if self.trigger_transition_check:
            self.buffered_observations_since_last_drift = self.cancel_trigger_val()
        if self.trigger_attempt_transition_check:
            self.buffered_observations_since_last_drift_attempt = self.cancel_trigger_val()

    def cancel_trigger_val(self):
        """ Shift the trigger past the trigger point, so it will not be called until it is reset.
        """
        return self.trigger_point + 1

    def reset_trigger_val(self):
        """ Reset a trigger to 0, so it will eventually be called.
        """
        return 0

    def _partial_fit(self, X, y, sample_weight, masked=False):

        # init defaults for trackers
        found_change = False
        self.detected_drift = False
        current_sensitivity = self.get_current_sensitivity()
        self.warning_detected = False
        self.observations_since_last_drift += 1

        # get_temporal_x, and get_imputed_label are to deal with masked
        # values where we don't see true label.
        # As the functions are now, they don't do anything extra.
        # But could be extended to reuse last made prediction as
        # the label for example.
        temporal_X = self.get_temporal_x(X)
        prediction = self.predict(temporal_X)[0]
        label = y if not masked else self.get_imputed_label(
            X=X, prediction=prediction, last_label=self.last_label)
        self.last_label = label
        # correctly_classified from the systems point of view.
        correctly_classifies = prediction == label

        current_observation = (temporal_X, label, prediction, correctly_classifies,
                               self.get_active_state().classifier.evolution, self.ex)
        # fingerprint_window holds window_size observations at the head of the stream.
        self.fingerprint_window.append(current_observation)
        # Add to buffers, and return clean data if availiable
        buffered_values, buffered_observation = self.update_and_check_buffer(
            current_observation)

        # Attempt to fit on possible clean data
        self.attempt_to_fit(buffered_values, sample_weight)

        # Detect if the fit changed the model in a way which
        # might significantly change behaviour.
        # If it did, we need to account for this in terms
        # of our fingerprint.
        self.check_evolution()
        self.monitor_evolution_status(buffered_observation)

        # Setup global vars handling the current partial fit
        self.reset_partial_fit_settings()

        take_measurement = self.check_take_measurement()
        if take_measurement:
            if is_valid_fingerprint(self.get_active_state().fingerprint):
                self.set_current_active_metainfo(add_to_normalizer=True)
                similarity = self.get_similarity_to_active_state(
                    self.active_metainfo, flat_nonorm_current_metainfo=self.active_nonormed_flat)
                self.update_similarity_change_detection(similarity)
                # For # logging
                self.monitor_active_state_active_similarity = similarity
            self.similarity_last_observation = self.ex

        do_update_fingerprint = self.check_do_update_fingerprint()
        if do_update_fingerprint:
            # We update fingerprints using the BUFFERED window
            # to make it less likely we are incorperating
            # a fingerprint from a transition region or a new
            # concept.
            # Downside is this is slightly behind, but with the
            # assumption that a concept is stationairy, we
            # should be able to handle slightly out of date
            # fingerprints.
            self.set_current_buffer_metainfo()
            self.get_active_state().update_fingerprint(
                self.buffered_metainfo, self.ex, normalizer=self.normalizer)
            self.fingerprint_changed_since_last_weight_calc = True
            # If we have seen a few fingerprints, so as to reduce variability,
            # we record this as a running 'normal' similarity.
            self.record_active_similarity()

        do_update_non_active_fingerprint = self.check_do_update_non_active_fingerprint()
        if do_update_non_active_fingerprint:
            self.set_current_buffer_metainfo()
            self.update_all_fingerprints_buffered()

        self.check_warning_detector()

        # Check main detector
        detected_drift = self.fingerprint_similarity_detector.detected_change()

        # If the main state trigers, or a previous detection is propagating due to lack of data,
        # trigger a change
        found_change = self.check_found_change(detected_drift)

        # Test similarity of each state to the head of the stream. This is purely for # logging and debugging purposes,
        # so can (and should) be disabled to test performance.
        take_active_observation = self.check_take_active_observation(
            found_change)
        if take_active_observation:
            self.monitor_all_fingerprints_active()

        # We have three reasons for attempting a check: We detected a drift, we want to evaluate the current state, or we have a propagating transition attempt.
        if found_change or self.trigger_transition_check or self.trigger_attempt_transition_check:
            self.attempt_transition()

        # Set exposed info for evaluation.
        self.active_state = self.active_state_id
        self.found_change = found_change
        self.states = self.state_repository
        self.current_sensitivity = current_sensitivity

    def rank_inactive_models_for_suitability_fingerprint(self):
        logging.info(
            f"Ranking inactive models on current stream at {self.ex}, using last {len(self.fingerprint_window)} elements.")
        if self.get_active_state().fingerprint is None:
            return None, None, None, False
        recent_window = self.fingerprint_window
        if len(recent_window) == 0:
            return [[], False, None, False]

        recent_X = np.vstack([x[0] for x in recent_window])[
            ::-1][:self.min_window_size]
        recent_labels = np.array([x[1] for x in recent_window])[
            ::-1][:self.min_window_size]

        shadow_model = self.learner()
        shadow_model.partial_fit(recent_X, recent_labels, classes=self.classes)

        state_performace_by_id = []
        filter_states = set()
        observation_accuracy = {}
        observation_stats = {}
        observation_fingerprints = {}
        observation_similarities = {}
        # Get the stats generated by the active model and
        # Update its fingerprint
        # DOn't add to normalizer as its already been added!
        self.set_current_active_metainfo(add_to_normalizer=False)
        active_accuracy = sum(
            [w[3] for w in self.fingerprint_window]) / len(self.fingerprint_window)
        fingerprint_feature_base = self.active_metainfo
        observation_accuracy["active"] = active_accuracy
        observation_stats["active"] = self.active_metainfo

        # Get the stats generated by each stored model
        # To do this, we need to reclassify using that model
        # to get correct meta-information such as error rate.
        # However we can reuse the feature data.
        for concept_id in self.state_repository:
            state = self.state_repository[concept_id]

            # Bit hacky. If a new concept is found for a transition,
            # we don't want to consider it for the subsequent check,
            # As it will always be similar as it has just trained on this
            # stuff. SO this is just for the double check transition.
            if state.seen < self.window_size * 2:
                continue
            if self.state_repository[concept_id].fingerprint is None:
                continue
            if concept_id != self.active_state_id:
                concept_classifier, concept_window, updated_predictions, updated_errors = self.make_window_predictions(
                    concept_id=concept_id, cache=self.all_states_active_cache, current_timestep=self.ex, observation_window=self.fingerprint_window)
                self.all_states_active_cache[concept_id] = (
                    updated_predictions, updated_errors, self.ex)

                concept_model_stats, concept_model_flat = self.get_metainfo_from_window(window=concept_window,
                                                                                        window_name=f"active_concept_{concept_id}",
                                                                                        buffered=False,
                                                                                        add_to_normalizer=True,
                                                                                        feature_base=fingerprint_feature_base,
                                                                                        feature_base_flat=self.active_nonormed_flat,
                                                                                        stored_shap=concept_classifier.shap_model)
                concept_model_accuracy = sum(
                    [w[3] for w in concept_window]) / len(concept_window)
            else:
                concept_model_accuracy = active_accuracy
                concept_model_stats = self.active_metainfo
                concept_model_flat = self.active_nonormed_flat

            observation_accuracy[concept_id] = concept_model_accuracy
            observation_stats[concept_id] = concept_model_stats

            similarity = self.get_similarity(
                concept_model_stats, state=concept_id, flat_nonorm_current_metainfo=concept_model_flat)
            observation_similarities[concept_id] = similarity
            if concept_id == self.active_state_id:
                observation_similarities["active"] = similarity

            logging.info(
                f"State {concept_id} performance: Similarity {similarity}")

            # We test to see if the accuracy on the current stream is similar to past uses
            # of a state. If not, we do not consider it.
            # We don't want to retest elements already tested in recent_window.
            if state.active_similarity_record is not None:
                def similarity_calc_func(x, y, z): return self.get_similarity(
                    x, state=concept_id, fingerprint_to_compare=y, flat_nonorm_current_metainfo=z)
                recent_similarity, normal_stdev = state.get_state_recent_similarity(
                    similarity_calc_func)
            else:
                continue
            logging.info(f"Current performance: Similarity {similarity}")
            logging.info(f"Recent performance: Similarity {recent_similarity}")

            recent_stdev = max(normal_stdev, self.similarity_min_stdev)
            recent_stdev = min(recent_stdev, self.similarity_max_stdev)

            standard_deviations_from_normal = 1 + \
                abs(similarity - recent_similarity) / recent_stdev
            logging.info(
                f"standard_deviations_from_normal: {standard_deviations_from_normal}")
            logging.info(f"Recent performance: stdev {recent_stdev}")
            logging.info(
                f"Similarity range: {recent_similarity - (self.similarity_num_stdevs * recent_stdev)} to {recent_similarity + (self.similarity_num_stdevs * recent_stdev)}")
            logging.info(
                f"Performace score: similarity: {1 - similarity}, adjusted by distance: {1 - (similarity * standard_deviations_from_normal)}")

            accept_state = True
            if similarity - recent_similarity > (self.similarity_num_stdevs * recent_stdev):
                accept_state = False
            if similarity < 0.001:
                accept_state = True
            if not accept_state:
                filter_states.add(concept_id)
                logging.info(f"State {concept_id} filtered")
            else:
                state_performace_by_id.append(
                    (concept_id, 1 - (similarity * standard_deviations_from_normal)))

        # State suitability in sorted (ascending) order
        state_performace_by_id.sort(key=lambda x: x[1])

        use_shadow = True
        if len(state_performace_by_id) > 0:
            use_shadow = False
            logging.info("Top state is similar")

        return [[x[0] for x in state_performace_by_id], use_shadow, shadow_model, True]

    def get_current_sensitivity(self):
        return self.base_sensitivity


def is_valid_fingerprint(fp):
    return fp is not None and fp.seen > 5
