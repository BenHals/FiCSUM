
import math
import warnings

import numpy as np
from ConceptFingerprint.Classifier.feature_selection.fisher_score import \
    fisher_score
from ConceptFingerprint.Classifier.feature_selection.mutual_information import *
from sklearn.feature_selection import mutual_info_regression


def check_fps_exist(fingerprints):
    for f in fingerprints:
        if f is None:
            print(fingerprints)
            raise ValueError(fingerprints)


def get_sources_and_features(fingerprints):
    """Get sources and features from fingerprints.
    Assume that all fingerprints are using the same format
    """
    sources = fingerprints[0].sources
    features = fingerprints[0].features
    return sources, features


def map_weights(sources, features, normalizer, fingerprints, state_active_non_active_fingerprints, weight_calc, mi_formula=None, weighted=False):
    """ Iterates over all features for all sources.
    Calculates a weight based on the passed function, and constructs a dictionary of [source][feature]: weight pairs.
    Also returns two flat vectors corresponding to the ordering defined in the normalizer, one including ignored keys and one excluding.
    An unordered but labeled version of the vector is also passed.
    """
    weights = {}
    flat_weight_vector = np.empty(normalizer.total_num_signals)
    ignore_flat_weight_vector = np.empty(normalizer.ignore_num_signals)
    labeled = []

    for s in sources:
        weights[s] = {}
        for f in features:
            flat_total_index = normalizer.total_indexes[s][f]
            flat_ignore_index = normalizer.ignore_indexes[s][f]

            weights[s][f] = weight_calc(
                s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector)

            flat_weight_vector[flat_total_index] = weights[s][f]
            ignore_flat_weight_vector[flat_ignore_index] = weights[s][f]
            labeled.append(
                (s, f, weights[s][f] if not np.isnan(weights[s][f]) else -1))

    return weights, flat_weight_vector, ignore_flat_weight_vector, labeled


def process_weights(normalizer, weights, flat_weight_vector, ignore_flat_weight_vector, labeled, nan_fill_val=0.01):
    """ Process the calculated weights, filling in nan values with the min value.
    """
    normalizer.changed_since_last_weight_calc = False
    min_val = np.nanmin(ignore_flat_weight_vector)
    mid_val = min_val
    if np.isnan(mid_val):
        mid_val = nan_fill_val
    ignore_flat_weight_vector = np.nan_to_num(
        ignore_flat_weight_vector, nan=mid_val)

    min_val = np.nanmin(flat_weight_vector)
    mid_val = min_val
    if np.isnan(mid_val):
        mid_val = nan_fill_val
    flat_weight_vector = np.nan_to_num(flat_weight_vector, nan=mid_val)
    sorted_weights = sorted(labeled, key=lambda x: x[2])
    return weights, flat_weight_vector, ignore_flat_weight_vector, sorted_weights


def weight_features(fingerprints, state_active_non_active_fingerprints, normalizer, state_id, calc_weight_func, nan_fill_val=0.01,  mi_formula=None, weighted=False):
    """ General process to weight each feature.
    """
    sources, features = get_sources_and_features(fingerprints)
    check_fps_exist(fingerprints)

    # Iterate over all features from all sources and call passed calc_weight_func to calculate a weight.
    # Returns four views of the weights
    weights, flat_weight_vector, ignore_flat_weight_vector, labeled = map_weights(
        sources, features, normalizer, fingerprints, state_active_non_active_fingerprints, calc_weight_func, mi_formula=mi_formula, weighted=weighted)

    # Process weight vectors, filling nan values etc
    weights, flat_weight_vector, ignore_flat_weight_vector, sorted_weights = process_weights(
        normalizer, weights, flat_weight_vector, ignore_flat_weight_vector, labeled, nan_fill_val=nan_fill_val)

    return weights, flat_weight_vector, ignore_flat_weight_vector, sorted_weights


def feature_selection_None(fingerprints, state_active_non_active_fingerprints, normalizer, state_id):
    """ Uniformly weights features
    """
    return weight_features(fingerprints, state_active_non_active_fingerprints, normalizer, state_id, lambda *args: 1.0)


def feature_selection_original(fingerprints, state_active_non_active_fingerprints, normalizer, state_id):
    """ Weight features according to an original algorithm, similar to fisher score.
    """
    return weight_features(fingerprints, state_active_non_active_fingerprints, normalizer, state_id, calc_weight_original, nan_fill_val=1)


def feature_selection_fisher_overall(fingerprints, state_active_non_active_fingerprints, normalizer, state_id):
    """ Weight features using the fisher score adjusted to be based on overall variance, rather than within concept variance
    """
    return weight_features(fingerprints, state_active_non_active_fingerprints, normalizer, state_id, calc_weight_fisher_overall)


def feature_selection_fisher(fingerprints, state_active_non_active_fingerprints, normalizer, state_id):
    """ Weight features using the standard fisher score
    """
    return weight_features(fingerprints, state_active_non_active_fingerprints, normalizer, state_id, calc_weight_fisher)


def feature_selection_MI(fingerprints, mi_formula, state_active_non_active_fingerprints, normalizer, state_id):
    """ Weight features using mutual information calculated from assuming a gaussian distribution
    given feature mean and standard deviation
    """
    return weight_features(fingerprints, state_active_non_active_fingerprints, normalizer, state_id, calc_weight_MI, mi_formula=mi_formula)


def feature_selection_cached_MI(fingerprints, mi_formula, state_active_non_active_fingerprints, normalizer, state_id):
    """ Weight features using mutual information calculated from assuming a gaussian distribution
    given feature mean and standard deviation. Uses cached values
    """
    return weight_features(fingerprints, state_active_non_active_fingerprints, normalizer, state_id, calc_weight_cached_MI, mi_formula=mi_formula)


def feature_selection_histogramMI(fingerprints, mi_formula, state_active_non_active_fingerprints, normalizer, state_id, weighted=False):
    """ Weight features using mutual information calculated from storing feature histograms
    """
    return weight_features(fingerprints, state_active_non_active_fingerprints, normalizer, state_id, calc_weight_histogramMI, mi_formula=mi_formula, weighted=weighted)


def feature_selection_histogram_covredMI(fingerprints, mi_formula, state_active_non_active_fingerprints, normalizer, state_id):
    """ Weight features using mutual information calculated from storing feature histograms. Incorporate redundancy though covariance.
    """
    return weight_features(fingerprints, state_active_non_active_fingerprints, normalizer, state_id, calc_weight_histogram_covredMI, mi_formula=mi_formula)


def calc_weight_original(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector):
    """ We want to weight dimensions inversely to their deviation.
    We want to weight dimensions with high deviation
    between concepts compared to within a concept.

    We min max scale vectors to remove raw location,
    then scale down such that the max intraconcept deviation is
    1% (0.01). This gives a standardised view to look at 
    inter-concept deviation.

    The final weight is the standard deviation of concept means
    in this scaled format. This takes into intra-concept deviation,
    as dimensions where this is large will be scaled more penalizing
    inter-concept deviation.
    """
    weight = None

    lower_bound, scaling_factor = normalizer.get_norm_scaling_factor(s, f)
    scale_factor = np.nan
    if len(fingerprints) < 2:
        if scaling_factor > 0:
            standard_deviations = [finger.fingerprint[s][f]["stdev"]
                                   for finger in fingerprints if finger.fingerprint[s][f]["stdev"] is not None]
            if len(standard_deviations) > 0:
                mean_stdev = np.mean(standard_deviations)
                mean_stdev = (mean_stdev / scaling_factor)
                if mean_stdev > 0:

                    scale_factor = 0.01 / max(mean_stdev, 0.01)
                    weight = scale_factor
                else:
                    weight = np.nan
            else:
                weight = np.nan
        else:
            weight = np.nan
    else:
        # Find max intra-concept stdev
        max_stdev = np.mean([finger.fingerprint[s][f]["stdev"]
                            for finger in fingerprints])
        if scaling_factor > 0 and max_stdev > 0:
            max_stdev = (max_stdev / scaling_factor)
            scale_factor = 0.01 / max(max_stdev, 0.01)
            values = np.array([finger.fingerprint_values[s][f]
                               for finger in fingerprints])
            max_min_scaled_values = (
                values - lower_bound) / scaling_factor

            intraconcept_scaled_values = max_min_scaled_values * scale_factor

            between_active_concept_stdev = np.std(
                intraconcept_scaled_values)

            between_segment_stdev = []
            for state_id in state_active_non_active_fingerprints:
                state_active_fp_values = state_active_non_active_fingerprints[
                    state_id][0].fingerprint_values
                state_non_active_fp_values = {
                    k: v.fingerprint_values for k, v in state_active_non_active_fingerprints[state_id][1].items()}
                values = np.array([state_active_fp_values[s][f], *[finger[s][f]
                                                                   for finger in state_non_active_fp_values.values()]])
                max_min_scaled_values = (
                    values - lower_bound) / scaling_factor
                intrasegment_scaled_values = max_min_scaled_values * scale_factor
                between_segment_concept_stdev = np.std(
                    intrasegment_scaled_values)
                between_segment_stdev.append(
                    between_segment_concept_stdev)

            feature_importance_weight = max(
                between_active_concept_stdev, np.mean(between_segment_stdev))
            weight = max(
                between_active_concept_stdev, np.mean(between_segment_stdev))
        else:
            weight = np.nan
    return weight


def calc_weight(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector, weight_func):
    """ General process for calculating a feature weight.
        Features are weighed independently.
        This does not capture covariance, so
        cannot find 'redundant' features!
        Features with high correlation should
        not be included in the set.
    """
    weight = None
    lower_bound, scaling_factor = normalizer.get_norm_scaling_factor(
        s, f)
    # If we only have seen one concept, we cannot work out how features change
    # across concepts. Our only data is the variance of each feature.
    # We scale each dimension such that 1 standard deviation is one unit.
    # (With a minimum stdev of 0.01)
    # Default values set to np.nan for 0 standard deviation.
    # These are replaced with the min seen weight.
    feature_importance_weight = np.nan
    scale_factor = np.nan

    # The scale factor describes the range of concept values.
    # We cannot weight a single value, so return a default
    # np.nan is processed out later
    if scaling_factor <= 0:
        return np.nan

    standard_deviations = [finger.fingerprint[s][f]["stdev"]
                           for finger in fingerprints if finger.fingerprint[s][f]["stdev"] is not None]

    if len(standard_deviations) == 0:
        return np.nan

    counts = np.array([finger.fingerprint[s][f]['seen']
                       for finger in fingerprints if finger.fingerprint[s][f]["seen"] is not None])
    mean_intraconcept_stdev = 0
    for count, stdev in zip(counts, standard_deviations):
        mean_intraconcept_stdev += count * (stdev / scaling_factor)
    mean_intraconcept_stdev /= np.sum(counts)

    # The scale factor scales each dimension so
    # that a deviance in each dimension is relatively
    # equivalent.
    # This is based on the average intra concept stdev,
    # i.e. the normal variance we would expect to see within
    # a fingerprint.
    # 1/stdev scales the dimension such that the unit distance
    # is equal to one standard deviation, i.e. deviations
    # used to calculate similarity are in terms of standard
    # deviation.
    # To constrain this weight to the range (0, 1], we
    # clamp stdev to [0.01, inf) and multiply by 0.01.
    # This transformation considers all stdevs under 0.01
    # to be the same 'small' value.
    # This ensures unstable features which sometimes stay
    # very similar then change a lot, e.g. FI don't get
    # huge weights from small stdevs.
    # The * 0.01 means we consider
    # the unit distance to be 100 standard deviations.
    # The base meaning is the same under this transformation.
    scale_factor = 0.01 / max(mean_intraconcept_stdev, 0.01)

    scaled_overall_stdev = (
        normalizer.data_stdev[s][f]["stdev"]) / scaling_factor
    
    #<TODO> CHECK WHY THIS IS STDEV!
    # scaled_overall_mean = (
    #         normalizer.data_stdev[s][f]["stdev"] - lower_bound) / scaling_factor
    scaled_overall_mean = (
            normalizer.data_stdev[s][f]["value"] - lower_bound) / scaling_factor

    if scaled_overall_stdev < 0:
        return np.nan

    # If the number of fingerprints is 1, we have not seen any concept drift.
    # We cannot identify which features will change, so just weight according to
    # standard deviation.
    if len(fingerprints) < 2:
        return scale_factor

    weight = weight_func(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula,
                         weighted, flat_weight_vector, lower_bound, scaling_factor, scale_factor, scaled_overall_stdev, scaled_overall_mean)
    return weight

def calc_weight_fisher_overall(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector):
    return calc_weight(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector, fisher_overall_weight)

def calc_weight_fisher(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector):
    return calc_weight(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector, fisher_weight)

def calc_weight_MI(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector):
    return calc_weight(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector, MI_weight)

def calc_weight_cached_MI(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector):
    return calc_weight(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector, cached_MI_weight)

def calc_weight_histogramMI(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector):
    return calc_weight(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector, histogramMI_weight)

def calc_weight_histogram_covredMI(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector):
    return calc_weight(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector, histogram_covredMI_weight)

def fisher_overall_weight(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector, lower_bound, scaling_factor, scale_factor, scaled_overall_stdev, scaled_overall_mean):
    # Feature importance is made up of 2 factors.
    # Using the classifier for concept C, distinguish active and non-active fingerprints.
    # And given the classifier and fingerprint from a set of concepts determine the active one.

    # We first calculate the fisher score between active-nonactive for all concepts,
    # and get the average.
    between_segment_fisher_scores = []
    for state_id in state_active_non_active_fingerprints:
        state_active_fp_values = state_active_non_active_fingerprints[state_id][0]
        state_non_active_fp_values = {
            k: v for k, v in state_active_non_active_fingerprints[state_id][1].items()}
        non_active_means = np.array([state_active_fp_values.fingerprint_values[s][f], *[
                                    finger.fingerprint_values[s][f] for finger in state_non_active_fp_values.values()]])
        non_active_scaled_means = (
            non_active_means - lower_bound) / scaling_factor
        non_active_counts = np.array([state_active_fp_values.fingerprint[s][f]['seen'],  *[
            finger.fingerprint[s][f]['seen'] for finger in state_non_active_fp_values.values()]])
        between_segment_fisher_score = fisher_score(
            non_active_scaled_means, non_active_counts, scaled_overall_stdev)
        between_segment_fisher_scores.append(
            between_segment_fisher_score)

    # We then calculate the fisher score between all active concepts.
    # Get standard deviations of each concept.
    standard_deviations = np.array(
        [finger.fingerprint[s][f]["stdev"] for finger in fingerprints if finger.fingerprint[s][f]["stdev"] is not None])
    scaled_standard_deviations = (
        standard_deviations) / scaling_factor
    # Get mean of each concept
    means = np.array([finger.fingerprint_values[s][f]
                      for finger in fingerprints])
    scaled_means = (means - lower_bound) / scaling_factor
    counts = np.array([finger.fingerprint[s][f]['seen']
                       for finger in fingerprints if finger.fingerprint[s][f]["seen"] is not None])
    between_active_fisher_score = fisher_score(
        scaled_means, counts, scaled_overall_stdev)

    feature_importance_weight = max(
        between_active_fisher_score, np.mean(between_segment_fisher_scores))
    weight = feature_importance_weight * \
        scale_factor
    return weight

def fisher_weight(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector, lower_bound, scaling_factor, scale_factor, scaled_overall_stdev, scaled_overall_mean):
    between_segment_fisher_scores = []
    for state_id in state_active_non_active_fingerprints:
        state_active_fp_values = state_active_non_active_fingerprints[state_id][0]
        state_non_active_fp_values = {
            k: v for k, v in state_active_non_active_fingerprints[state_id][1].items()}
        non_active_means = np.array([state_active_fp_values.fingerprint_values[s][f], *[
                                    finger.fingerprint_values[s][f] for finger in state_non_active_fp_values.values()]])
        non_active_stdev = np.array([state_active_fp_values.fingerprint[s][f]['stdev'], *[
                                    finger.fingerprint[s][f]['stdev'] for finger in state_non_active_fp_values.values()]])
        non_active_scaled_means = (
            non_active_means - lower_bound) / scaling_factor
        non_active_scaled_stdev = (
            non_active_stdev) / scaling_factor
        non_active_counts = np.array([state_active_fp_values.fingerprint[s][f]['seen'],  *[
            finger.fingerprint[s][f]['seen'] for finger in state_non_active_fp_values.values()]])
        between_segment_fisher_score = fisher_score(
            non_active_scaled_means, non_active_counts, scaled_overall_stdev, non_active_scaled_stdev)
        between_segment_fisher_scores.append(
            between_segment_fisher_score)

    # We then calculate the fisher score between all active concepts.
    # Get standard deviations of each concept.
    standard_deviations = np.array(
        [finger.fingerprint[s][f]["stdev"] for finger in fingerprints if finger.fingerprint[s][f]["stdev"] is not None])
    scaled_standard_deviations = (
        standard_deviations) / scaling_factor
    # Get mean of each concept
    means = np.array([finger.fingerprint_values[s][f]
                        for finger in fingerprints])
    scaled_means = (means - lower_bound) / scaling_factor
    counts = np.array([finger.fingerprint[s][f]['seen']
                        for finger in fingerprints if finger.fingerprint[s][f]["seen"] is not None])
    between_active_fisher_score = fisher_score(
        scaled_means, counts, scaled_overall_stdev, scaled_standard_deviations)

    feature_importance_weight = max(
        between_active_fisher_score, np.mean(between_segment_fisher_scores))
    weight = feature_importance_weight * \
        scale_factor
    return weight

def MI_weight(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector, lower_bound, scaling_factor, scale_factor, scaled_overall_stdev, scaled_overall_mean):
    between_segment_MIMs = []
    for state_id in state_active_non_active_fingerprints:
        state_active_fp_values = state_active_non_active_fingerprints[state_id][0]
        state_non_active_fp_values = {
            k: v for k, v in state_active_non_active_fingerprints[state_id][1].items()}
        non_active_means = np.array([state_active_fp_values.fingerprint_values[s][f], *[
                                    finger.fingerprint_values[s][f] for finger in state_non_active_fp_values.values()]])
        non_active_scaled_means = (
            non_active_means - lower_bound) / scaling_factor
        non_active_counts = np.array([state_active_fp_values.fingerprint[s][f]['seen'],  *[
            finger.fingerprint[s][f]['seen'] for finger in state_non_active_fp_values.values()]])
        non_active_stdevs = np.array([state_active_fp_values.fingerprint[s][f]['stdev'],  *[
            finger.fingerprint[s][f]['stdev'] for finger in state_non_active_fp_values.values()]])
        scaled_stdevs = (
            non_active_stdevs) / scaling_factor
        # We take the range as the max/min seen in fingerprints. This is NOT neccesarily the overall range seen in the normalizer!
        # this is because some observations are in boundary states and do not appear in a fingerprints.
        # These should be discarded as they are often outliers.
        concept_fingerprints = [
            state_active_fp_values, *[finger for finger in state_non_active_fp_values.values()]]
        min_val = min([finger.normalizer.data_ranges[s][f][0]
                        for finger in concept_fingerprints])
        max_val = max([finger.normalizer.data_ranges[s][f][1]
                        for finger in concept_fingerprints])
        range_start = (
            min_val - lower_bound) / scaling_factor
        range_end = (max_val - lower_bound) / \
            scaling_factor
        between_segment_MIM = mi_formula(scaled_overall_mean, scaled_overall_stdev, [
            range_start, range_end], non_active_scaled_means, scaled_stdevs, non_active_counts)
        between_segment_MIMs.append(between_segment_MIM)

    # We then calculate the fisher score between all active concepts.
    # Get standard deviations of each concept.
    standard_deviations = np.array(
        [finger.fingerprint[s][f]["stdev"] for finger in fingerprints if finger.fingerprint[s][f]["stdev"] is not None])
    scaled_standard_deviations = (
        standard_deviations) / scaling_factor
    # Get mean of each concept
    means = np.array([finger.fingerprint_values[s][f]
                        for finger in fingerprints])
    scaled_means = (means - lower_bound) / scaling_factor
    counts = np.array([finger.fingerprint[s][f]['seen']
                        for finger in fingerprints if finger.fingerprint[s][f]["seen"] is not None])
    concept_fingerprints = [
        finger for finger in fingerprints]
    # As above for range.
    range_start = (min([finger.normalizer.data_ranges[s][f][0]
                        for finger in concept_fingerprints]) - lower_bound) / scaling_factor
    range_end = (max([finger.normalizer.data_ranges[s][f][1]
                        for finger in concept_fingerprints]) - lower_bound) / scaling_factor
    between_active_MIM = mi_formula(scaled_overall_mean, scaled_overall_stdev, [
                                    range_start, range_end], scaled_means, scaled_standard_deviations, counts)

    # The scale factor scales the dimension such that a 1 stdev movement * weight = 1.
    # This lets movements be compared on their relative deviation from normal.
    # scale_factor = 1 / max(np.mean(scaled_standard_deviations), 0.1)
    weight = max(between_active_MIM, np.mean(
        between_segment_MIMs)) * scale_factor
    
    return weight

def cached_MI_weight(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector, lower_bound, scaling_factor, scale_factor, scaled_overall_stdev, scaled_overall_mean):
    between_segment_MIMs = []
    for state_id in state_active_non_active_fingerprints:
        state_active_fp_values = state_active_non_active_fingerprints[state_id][0]
        state_non_active_fp_values = {
            k: v for k, v in state_active_non_active_fingerprints[state_id][1].items()}
        non_active_means = np.array([state_active_fp_values.fingerprint_values[s][f], *[
                                    finger.fingerprint_values[s][f] for finger in state_non_active_fp_values.values()]])
        non_active_scaled_means = (
            non_active_means - lower_bound) / scaling_factor
        non_active_counts = np.array([state_active_fp_values.fingerprint[s][f]['seen'],  *[
            finger.fingerprint[s][f]['seen'] for finger in state_non_active_fp_values.values()]])
        non_active_stdevs = np.array([state_active_fp_values.fingerprint[s][f]['stdev'],  *[
            finger.fingerprint[s][f]['stdev'] for finger in state_non_active_fp_values.values()]])
        scaled_stdevs = (
            non_active_stdevs) / scaling_factor

        concept_fingerprints = [
            state_active_fp_values, *[finger for finger in state_non_active_fp_values.values()]]

        between_segment_MIM, _, _, _ = mi_formula(
            scaled_overall_mean, scaled_overall_stdev, concept_fingerprints, s, f)
        between_segment_MIMs.append(between_segment_MIM)

    # We then calculate the fisher score between all active concepts.
    # Get standard deviations of each concept.
    standard_deviations = np.array(
        [finger.fingerprint[s][f]["stdev"] for finger in fingerprints if finger.fingerprint[s][f]["stdev"] is not None])
    scaled_standard_deviations = (
        standard_deviations) / scaling_factor
    # Get mean of each concept
    means = np.array([finger.fingerprint_values[s][f]
                        for finger in fingerprints])
    scaled_means = (means - lower_bound) / scaling_factor
    counts = np.array([finger.fingerprint[s][f]['seen']
                        for finger in fingerprints if finger.fingerprint[s][f]["seen"] is not None])

    concept_fingerprints = [
        finger for finger in fingerprints]
    between_active_MIM, _, _, _ = mi_formula(
        scaled_overall_mean, scaled_overall_stdev, concept_fingerprints, s, f)

    # The scale factor scales the dimension such that a 1 stdev movement * weight = 1.
    # This lets movements be compared on their relative deviation from normal.
    # scale_factor = 1 / max(np.mean(scaled_standard_deviations), 0.1)
    weight = max(between_active_MIM, np.mean(
        between_segment_MIMs)) * scale_factor
    
    return weight

def histogramMI_weight(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector, lower_bound, scaling_factor, scale_factor, scaled_overall_stdev, scaled_overall_mean):
    between_segment_MIMs = []
    merged_histogram, merged_bins, merged_count = None, None, None
    for state_id in state_active_non_active_fingerprints:
        state_active_fp_values = state_active_non_active_fingerprints[state_id][0]
        state_non_active_fp_values = {
            k: v for k, v in state_active_non_active_fingerprints[state_id][1].items()}
        # between_segment_MIM, merged_histogram, merged_bins, merged_count = mi_formula(normalizer, [state_active_fp_values, *list(state_non_active_fp_values.values())], s, f, merged_histogram, merged_bins, merged_count)
        # Can't resuse merged histograms! They are different for each run.
        if not weighted:
            between_segment_MIM, merged_histogram, merged_bins, merged_count = mi_formula(
                normalizer, [state_active_fp_values, *list(state_non_active_fp_values.values())], s, f)
        else:
            between_segment_MIM, merged_histogram, merged_bins, merged_count = mi_formula(
                normalizer, [state_active_fp_values, *list(state_non_active_fp_values.values())], s, f, weighted=True, weights=flat_weight_vector)
        between_segment_MIMs.append(between_segment_MIM)

    if not weighted:
        between_active_MIM, _, _, _ = mi_formula(
            normalizer, fingerprints, s, f)
    else:
        between_active_MIM, _, _, _ = mi_formula(
            normalizer, fingerprints, s, f, weighted=True, weights=flat_weight_vector)

    weight = max(between_active_MIM, np.mean(
        between_segment_MIMs)) * scale_factor
    
    return weight

def histogram_covredMI_weight(s, f, normalizer, fingerprints, state_active_non_active_fingerprints, mi_formula, weighted, flat_weight_vector, lower_bound, scaling_factor, scale_factor, scaled_overall_stdev, scaled_overall_mean):
    flat_ignore_index = normalizer.ignore_indexes[s][f]
    between_segment_MIMs = []
    merged_histogram, merged_bins, merged_count = None, None, None
    for state_id in state_active_non_active_fingerprints:
        state_active_fp_values = state_active_non_active_fingerprints[state_id][0]
        state_non_active_fp_values = {
            k: v for k, v in state_active_non_active_fingerprints[state_id][1].items()}
        # between_segment_MIM, merged_histogram, merged_bins, merged_count = mi_formula(normalizer, [state_active_fp_values, *list(state_non_active_fp_values.values())], s, f, merged_histogram, merged_bins, merged_count)
        # Can't resuse merged histograms! They are different for each run.
        concept_fingerprints = [
            state_active_fp_values, *list(state_non_active_fp_values.values())]
        between_segment_MIM, merged_histogram, merged_bins, merged_count = mi_formula(
            normalizer, concept_fingerprints, s, f)
        overall_pairwise_redundancy, class_conditional_redundancy = calculate_JR_MRMR_sketch(
            concept_fingerprints, flat_ignore_index)
        between_segment_MIM = max(
            between_segment_MIM - overall_pairwise_redundancy - class_conditional_redundancy, 0)
        between_segment_MIMs.append(between_segment_MIM)

    overall_pairwise_redundancy, class_conditional_redundancy = calculate_JR_MRMR_sketch(
        fingerprints, flat_ignore_index)
    between_active_MIM, _, _, _ = mi_formula(
        normalizer, fingerprints, s, f)
    between_active_MIM = max(
        between_active_MIM - overall_pairwise_redundancy - class_conditional_redundancy, 0)

    weight = max(between_active_MIM, np.mean(
        between_segment_MIMs)) * scale_factor

    return weight

def mi_from_fingerprint_sketch(normalizer, concept_fingerprints, s, f, merged_histogram=None, merged_bins=None, merged_count=None):

    ranges = []
    for fingerprint in concept_fingerprints:
        ranges.append(fingerprint.get_feature_range(s, f))
    min_val = min([x[0] for x in ranges])
    max_val = max([x[1] for x in ranges])
    num_bins = concept_fingerprints[0].num_bins
    np_bins = make_bins((min_val, max_val), num_bins)

    concept_histograms = []
    concept_bins = []
    concept_counts = []
    all_values = []
    all_count = 0
    total_weight = 0.0
    for fingerprint in concept_fingerprints:
        c_histogram_np, c_bins_np, c_values_np = fingerprint.get_binned_feature(
            s, f, np_bins, np_bins=np_bins)
        c_count = fingerprint.num_bins
        concept_histograms.append(c_histogram_np)
        concept_bins.append(c_bins_np)
        concept_counts.append(c_count)
        total_weight += c_count
        all_count += c_count
        all_values.append(c_values_np)
    all_values = np.concatenate(all_values)
    if not merged_histogram:
        merged_histogram = bin_values(all_values, np_bins)
        merged_bins = np_bins
        merged_count = all_count

    concept_histograms = np.vstack(concept_histograms)
    concept_bins = np.vstack(concept_bins)

    m_entropy = MI_histogram_estimation_cache_mat(
        merged_histogram, merged_bins, merged_count, concept_histograms, concept_bins, concept_counts, total_weight)
    return m_entropy, merged_histogram, merged_bins, merged_count


def mi_cov_from_fingerprint_sketch(normalizer, concept_fingerprints, s, f, merged_histogram=None, merged_bins=None, merged_count=None, weighted=False, weights=None):

    ranges = []
    for fingerprint in concept_fingerprints:
        ranges.append(fingerprint.get_feature_range(s, f))
    min_val = min([x[0] for x in ranges])
    max_val = max([x[1] for x in ranges])
    num_bins = concept_fingerprints[0].num_bins
    np_bins = make_bins((min_val, max_val), num_bins)

    concept_histograms = []
    concept_bins = []
    concept_counts = []
    all_values = []
    absolute_correlations = []
    all_count = 0
    total_weight = 0.0
    for fingerprint in concept_fingerprints:
        if type(fingerprint) is int:
            print(concept_fingerprints)
            print(fingerprint)
        c_histogram_np, c_bins_np, c_values_np = fingerprint.get_binned_feature(
            s, f, np_bins, np_bins=np_bins)
        feature_selected_covariance, feature_selected_correlation, feature_selected_abs_correlation = fingerprint.get_binned_covariance(
            s, f)
        absolute_correlations.append(feature_selected_abs_correlation)
        c_count = fingerprint.num_bins
        concept_histograms.append(c_histogram_np)
        concept_bins.append(c_bins_np)
        concept_counts.append(c_count)
        total_weight += c_count
        all_count += c_count
        all_values.append(c_values_np)
    all_values = np.concatenate(all_values)
    if not merged_histogram:
        merged_histogram = bin_values(all_values, np_bins)
        merged_bins = np_bins
        merged_count = all_count

    concept_histograms = np.vstack(concept_histograms)
    concept_bins = np.vstack(concept_bins)
    absolute_correlations = np.vstack(absolute_correlations)
    concept_mean_abs_corr = np.mean(absolute_correlations, axis=0)
    if weighted:
        concept_mean_abs_corr = concept_mean_abs_corr * \
            weights[:concept_mean_abs_corr.shape[0]]
    try:
        corr_sum = np.mean(concept_mean_abs_corr)
    except Exception as e:
        print(e)
        raise(e)
    max_corr = 1
    # How much below max correlation we are,
    # in range [0, 1].
    # 1 indicates 0 correlation, so no penalty
    # 0 indicates max correlation, so fully penalty
    correlation_avoided = max_corr - corr_sum

    m_entropy = MI_histogram_estimation_cache_mat(
        merged_histogram, merged_bins, merged_count, concept_histograms, concept_bins, concept_counts, total_weight)
    entropy_over_corr = m_entropy * correlation_avoided
    return entropy_over_corr, merged_histogram, merged_bins, merged_count


def calculate_JR_MRMR_sketch(concept_fingerprints, feature_index):
    """ Given feature index i, we wish to calculate the average redundancy between
    x_i and all x_j where j < i.
    Redundancy comes in two forms (for mutual information), pairwise independence and
    class conditional pairwise independence.
    So we need to calculate I(x_i ; x_j) and I(x_i ; x_j | Y).
    In otherwords, mutual info for all classes (here across all concepts) and average mutual info
    across each concept.
    To calculate mutual info, we get the vector from sketches (concat across all classes for overall) for x_i
    and x_j, and use sklearn mutual_info classifier to calculate mutual info.
    """

    pairwise_sum = 0.0
    cc_pairwise_sum = 0.0
    for j in range(feature_index):
        overall_i_values = []
        overall_j_values = []
        concept_pairwise_sum = 0.0
        concept_total_weight = 0.0
        for y_concept in concept_fingerprints:
            i_values = y_concept.sketch.get_column(feature_index)
            overall_i_values.append(i_values)
            j_values = y_concept.sketch.get_column(j)
            overall_j_values.append(j_values)
            class_infogain = 0.0
            concept_total_weight += y_concept.seen
            concept_pairwise_sum += class_infogain * y_concept.seen
        concept_pairwise_val = concept_pairwise_sum / concept_total_weight
        cc_pairwise_sum += concept_pairwise_val

        overall_infogain = mutual_info_regression(np.concatenate(
            overall_i_values).reshape(-1, 1), np.concatenate(overall_j_values), n_neighbors=1)[0]
        pairwise_sum += overall_infogain
    return pairwise_sum / feature_index, cc_pairwise_sum / feature_index

def make_bins(bin_range, num_bins):
    """ Construct array defining numpy binning given a range and number of bins.
    """
    total_width = bin_range[1] - bin_range[0]
    bin_width = total_width / num_bins
    new_np_bins = np.empty(num_bins + 1)
    for b_i in range(num_bins):
        i = bin_range[0] + b_i * bin_width
        new_np_bins[b_i] = i
    new_np_bins[num_bins] = bin_range[1]

    return new_np_bins

def mi_from_fingerprint_histogram_cache(normalizer, concept_fingerprints, s, f, merged_histogram=None, merged_bins=None, merged_count=None):
    """ Calculate the Mutual Information (MI) between a meta-feature [s][f] and fingerprints, based on value histograms.
    First extracts the cached histograms stored in the fingerprints.
    Then constructs an overall histograms by mergeing all histograms into one.
    Then arranges all histograms on a consistent scale so MI can be calculated.
    Finally, calculates MI as the reduction in entropy gained from separating the overall histogram into the individual concept histograms.
    """

    # Extract concept histograms from the fingerprints
    concept_histograms = []
    concept_bins = []
    concept_bins_np = []
    concept_counts = np.empty(len(concept_fingerprints))
    total_weight = 0.0
    for fi, fingerprint in enumerate(concept_fingerprints):
        c_histogram = fingerprint.fingerprint[s][f]["Histogram"]
        c_bins = fingerprint.fingerprint[s][f]["Bins"]
        c_bins_np = fingerprint.fingerprint[s][f]["BinsNP"]
        c_count = math.ceil(fingerprint.fingerprint[s][f]["seen"])
        total_weight += c_count
        concept_histograms.append(np.array(c_histogram))
        concept_bins.append(c_bins)
        concept_bins_np.append(c_bins_np)
        concept_counts[fi] = c_count
    concept_histograms = np.vstack(concept_histograms)
    concept_bins_np = np.vstack(concept_bins_np)

    # Construct overall histogram, and rescale all to be on this range.
    merged_histogram_np, merged_bins_np = merge_histograms_np2(
        concept_histograms, concept_bins_np)

    scaled_concept_histograms = []
    scaled_concept_bins = []
    scaled_concept_counts = np.empty(len(concept_fingerprints))
    for fi, fingerprint in enumerate(concept_fingerprints):
        c_histogram, c_bins = fingerprint.get_binned_feature(
            s, f, merged_bins_np, np_bins=merged_bins_np)
        c_count = math.ceil(fingerprint.fingerprint[s][f]["seen"])
        scaled_concept_histograms.append(np.array(c_histogram))
        scaled_concept_bins.append(c_bins)
        scaled_concept_counts[fi] = c_count
    scaled_concept_histograms = np.vstack(scaled_concept_histograms)

    # Calculate information gain from splitting the overall histogram.
    m_entropy = MI_histogram_estimation_cache_mat(
        merged_histogram_np, merged_bins_np, total_weight, scaled_concept_histograms, scaled_concept_bins, scaled_concept_counts, total_weight)
    return m_entropy, merged_histogram, merged_bins, merged_count


def mi_from_cached_fingerprint_bins(scaled_mean, scaled_stdev, concept_fingerprints, s, f, merged_histogram=None, merged_bins=None, merged_count=None):
    """ Calculate mutual information by approximating each feature distribution as a gaussian.
    We construct a virtual histogram from this distribution, which can be used to calculate mutual information.
    """
    concept_counts, counts_y_given_x, bin_weights, total_weight = get_bins_from_fingerprints_fast(
        concept_fingerprints, s, f)
    mat_entropy = MI_estimation_cache_mat(
        concept_counts, counts_y_given_x, bin_weights, total_weight)
    return mat_entropy, merged_histogram, merged_bins, merged_count
