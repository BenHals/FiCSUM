""" Functions for calculating feature importance via Mutual Information.
The basic idea is that a feature has more importance if it has higher mutual information
with the label, i.e, knowing the feature values tells us more about the label values.
Mutual information is related to Information Gain and KL divergence.
Mutual information = I(X; Y)
    = H(X) - H(X|Y)
    = H(Y) - H(Y|X)
    = H(X) + H(Y) - H(X, Y)
    = H(X, Y) - H(X|Y) - H(Y|X)

Mutual information between X and Y is the same as the information gain.
This is equal to the entropy difference between Y and Y | X, i.e. the loss 
in entropy in the label if we know X.

KL divergence measures the difference of two distributions. 
I(X; Y) = difference between joint (p(X, Y)) and marginal (p(X)p(Y)) 
    distributions
    = DKL(P(X, Y) || p(X)p(Y))
    or 
    expected difference between p(X|Y) and p(X) across all Y=y
    =Ey[DKL(p(X|Y=y) || p(X))]

Calculating these values if data is known is relatively easy.
Calculating online is harder, as we cannot store past data.
We need to create a constant space approximation of the distribution
of X and Y.
"""
import math
import numpy as np
import scipy.stats
import sys
from numba import jit, njit
from numba.core import types
from numba.typed import Dict, List
# from julia.api import Julia
# jl = Julia()
# from julia import Main
# jl.eval('include("ConceptFingerprint\\\\Classifier\\\\feature_selection\\\\mutual_information.jl")')


def normal_distribution_entropy(mu, sigma):
    return 0.5 * math.log2(2*math.pi*math.e*math.pow(sigma, 2))


def uniform_distribution_entropy(mu, sigma):
    return math.log2((mu+(sigma*2)) - (mu-(sigma*2)))


def clamp_sigma(sigma):
    # return max(sigma, 0.01)
    return sigma


def information_gain_normal_distributions(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    """ Constant space approximation of the distribution of X and Y as normal distributions.
    We approximate p(X) as a normal distribution for which H(X) has a closed form.
    We also approximate p(X|Y) as normal, and calculate H(X|Y) as Ey(H(X|Y=y)).
    I(X; Y) is then H(X) - H(X|Y)
    Problem: even if p(X|Y=y) is approx normal, i.e., feature is normal per concept,
    p(X) is unlikely to be normal overall! More likely this is a gaussian mixture which
    does not have a closed form entropy calculation. 
    We see empirically a large difference between true sampled entropy using this approximation. 
    We normalize based on the maximum possible entropy.
    """
    conditional_entropy = 0
    total_count = 0
    for count, sigma in zip(post_split_counts, post_split_sigmas):
        if sigma > 0:
            conditional_entropy += count * \
                normal_distribution_entropy(0, clamp_sigma(sigma))
            total_count += count
    conditional_entropy /= np.sum(total_count)
    initial_entropy = normal_distribution_entropy(
        0, clamp_sigma(initial_sigma))
    max_initial_entropy = normal_distribution_entropy(
        0, clamp_sigma(initial_sigma))
    normalized_initial = initial_entropy/max_initial_entropy
    max_conditional = normal_distribution_entropy(
        0, clamp_sigma(initial_sigma))
    normalized_conditional = conditional_entropy / max_conditional
    return normalized_initial - normalized_conditional


def information_gain_normal_distributions_UN(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    conditional_entropy = 0
    total_count = 0
    for count, sigma in zip(post_split_counts, post_split_sigmas):
        if sigma > 0:
            conditional_entropy += count * \
                normal_distribution_entropy(0, clamp_sigma(sigma))
            total_count += count
    conditional_entropy /= np.sum(total_count)
    initial_entropy = normal_distribution_entropy(
        0, clamp_sigma(initial_sigma))
    max_initial_entropy = normal_distribution_entropy(
        0, clamp_sigma(initial_sigma))
    normalized_initial = initial_entropy/max_initial_entropy
    max_conditional = normal_distribution_entropy(
        0, clamp_sigma(initial_sigma))
    normalized_conditional = conditional_entropy / max_conditional
    return initial_entropy - conditional_entropy


def information_gain_normal_distributions_Hx(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    """ Constant space approximation of the distribution of X and X|Y as bins and normal distributions.
    We approximate p(X) using bins.
    We approximate p(X|Y) as normal, and calculate H(X|Y) as Ey(H(X|Y=y)).
    I(X; Y) is then H(X) - H(X|Y)
    We see empirically a large difference between true sampled entropy using this approximation. 
    We normalize based on the maximum possible entropy.
    """
    conditional_entropy = 0
    total_count = 0
    for count, sigma in zip(post_split_counts, post_split_sigmas):
        if sigma > 0:
            conditional_entropy += count * \
                normal_distribution_entropy(0, clamp_sigma(sigma))
            total_count += count
    conditional_entropy /= np.sum(total_count)
    initial_entropy = estimate_Hx(initial_mu, initial_sigma, initial_range,
                                  post_split_mus, post_split_sigmas, post_split_counts, num_bins)
    max_initial_entropy = math.log2(num_bins)
    normalized_initial = initial_entropy/max_initial_entropy
    max_conditional = normal_distribution_entropy(
        0, clamp_sigma(initial_sigma))
    normalized_conditional = conditional_entropy / max_conditional
    return normalized_initial - normalized_conditional


def information_gain_normal_distributions_HxCond(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    """ Constant space approximation of the distribution of X and X|Y as bins.
    We approximate p(X) using bins.
    We approximate p(X|Y) as bins, and calculate H(X|Y) as Ey(H(X|Y=y)).
    I(X; Y) is then H(X) - H(X|Y)
    We see empirically a this is equiv to sampled IG.
    We scale to the maximum possible entropy
    """
    initial_entropy, conditional_entropies = estimate_Hx_cond(
        initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins)
    conditional_entropy = 0
    total_count = 0
    for count, e in zip(post_split_counts, conditional_entropies):
        conditional_entropy += count * e
        total_count += count
    conditional_entropy /= np.sum(total_count)
    max_initial_entropy = math.log2(num_bins)
    normalized_initial = initial_entropy/max_initial_entropy
    max_conditional = math.log2(num_bins)
    normalized_conditional = conditional_entropy / max_conditional
    return normalized_initial - normalized_conditional


def information_gain_normal_distributions_HxCond_UN(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    initial_entropy, conditional_entropies = estimate_Hx_cond(
        initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins)
    conditional_entropy = 0
    total_count = 0
    for count, e in zip(post_split_counts, conditional_entropies):
        conditional_entropy += count * e
        total_count += count
    conditional_entropy /= np.sum(total_count)
    max_initial_entropy = math.log2(num_bins)
    normalized_initial = initial_entropy/max_initial_entropy
    max_conditional = math.log2(num_bins)
    normalized_conditional = conditional_entropy / max_conditional
    return initial_entropy - conditional_entropy


def information_gain_normal_distributions_JS(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    """ Constant space approximation of the distribution of X and X|Y using jacard similarity.
    We approximate p(X) using bins.
    We approximate p(X|Y) as normal, and calculate H(X|Y) as Ey(H(X|Y=y)).
    I(X; Y) is then H(X) - H(X|Y)
    We see empirically a large difference between true sampled entropy using this approximation. 
    We normalize based on the maximum possible entropy.
    """
    summed_mu = 0
    summed_variance = 0
    total_count = sum(
        [c for c, s in zip(post_split_counts, post_split_sigmas) if s > 0])
    for sigma, count in zip(post_split_sigmas, post_split_counts):
        if sigma <= 0:
            continue
        freq = count / total_count
        variance = math.pow(freq*sigma, 2)
        summed_variance += variance
    summed_sigma = math.sqrt(summed_variance)

    initial_entropy = normal_distribution_entropy(summed_mu, summed_sigma)
    conditional_entropy = 0
    total_count = 0
    for count, sigma in zip(post_split_counts, post_split_sigmas):
        if sigma > 0:
            conditional_entropy += count * \
                normal_distribution_entropy(0, clamp_sigma(sigma))
            total_count += count
    conditional_entropy /= np.sum(total_count)

    return initial_entropy - conditional_entropy


def information_gain_normal_distributions_uniform(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    """ Constant space approximation of the distribution of X and X|Y as uniform distributions.
    We approximate p(X) as uniform across the observed range.
    We approximate p(X|Y) as uniform, and calculate H(X|Y) as Ey(H(X|Y=y)).
    I(X; Y) is then H(X) - H(X|Y)
    We see empirically a large difference between true sampled entropy using this approximation. 
    We normalize based on the maximum possible entropy.
    """
    conditional_entropy = 0
    total_count = 0
    for count, sigma in zip(post_split_counts, post_split_sigmas):
        if sigma > 0:
            conditional_entropy += count * \
                uniform_distribution_entropy(0, clamp_sigma(sigma))
            total_count += count
    conditional_entropy /= np.sum(total_count)
    return uniform_distribution_entropy(0, clamp_sigma(initial_sigma)) - conditional_entropy


def information_gain_normal_distributions_KL(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    """ Constant space approximation of the distribution of X and X|Y as normal.
    We approximate KL divergence using assumed normal distributions.
    We see empirically a large difference between true sampled entropy using this approximation. 
    We normalize based on the maximum possible entropy.
    """
    conditional_entropy = 0
    total_count = 0
    for count, mu, sigma in zip(post_split_counts, post_split_mus, post_split_sigmas):
        if sigma > 0:
            conditional_entropy += count * \
                KL_divergence(mu, clamp_sigma(sigma), initial_mu,
                              clamp_sigma(initial_sigma))
            total_count += count
    if total_count > 0:
        conditional_entropy /= total_count
    return conditional_entropy


def information_gain_normal_distributions_swap(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    """ Constant space approximation of the distribution of X and X|Y as normal.
    We approximate KL divergence using assumed normal distributions.
    Swapped order of KL divergence.
    We see empirically a large difference between true sampled entropy using this approximation. 
    We normalize based on the maximum possible entropy.
    """
    conditional_entropy = 0
    total_count = 0
    for count, mu, sigma in zip(post_split_counts, post_split_mus, post_split_sigmas):
        if sigma > 0:
            conditional_entropy += count * \
                KL_divergence(initial_mu, clamp_sigma(
                    initial_sigma), mu, clamp_sigma(sigma))
            total_count += count
    if total_count > 0:
        conditional_entropy /= total_count
    return conditional_entropy


def information_gain_normal_distributions_sym(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    return information_gain_normal_distributions_swap(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins) + information_gain_normal_distributions_KL(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins)


def KL_divergence(mu_A, sigma_A, mu_B, sigma_B):
    div = math.log2(sigma_B/sigma_A) + (math.pow(sigma_A, 2) +
                                        math.pow(mu_A - mu_B, 2))/(2 * math.pow(sigma_B, 2)) - 0.5
    return div


def estimate_Hy(class_counts):
    """ Rearrange standard entropy calculation for discrete var Y as used by scikit-multiflow
    """
    entropy = 0.0
    dis_sums = 0.0
    for i, count in enumerate(class_counts):
        if count > 0.0:
            entropy -= count * np.log2(count)
            dis_sums += count
    return (entropy + (dis_sums * np.log2(dis_sums))) / dis_sums if dis_sums > 0.0 else 0.0


def estimate_Hy_np(class_counts):
    """ Rearrange standard entropy calculation for discrete var Y as used by scikit-multiflow
    """
    positive_counts = class_counts[class_counts > 0]
    positive_probabilities = positive_counts / np.sum(positive_counts)
    log_probabilities = np.log2(positive_probabilities)
    # for i, count in enumerate(class_counts):
    #     if count > 0.0:
    #         entropy -= count * np.log2(count)
    #         dis_sums += count
    # return (entropy + (dis_sums * np.log2(dis_sums))) / dis_sums if dis_sums > 0.0 else 0.0
    entropy = -1 * (np.dot(positive_probabilities, log_probabilities))
    # dis_sums = np.sum(positive_counts)
    # return (entropy + (dis_sums * np.log2(dis_sums))) / dis_sums if dis_sums > 0.0 else 0.0
    return entropy


def estimate_Hx(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    """ Estimate the entropy of a feature X given normal distributions
    collected from K concepts.
    """
    counts_y_given_x, concept_counts = bin_concept_counts(
        initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins)
    entropy = 0.0
    dis_sums = 0.0
    for bin_class_counts in counts_y_given_x:
        count = np.sum(bin_class_counts)
        if count > 0.0:
            entropy -= count * np.log2(count)
            dis_sums += count
    return (entropy + (dis_sums * np.log2(dis_sums))) / dis_sums if dis_sums > 0.0 else 0.0


def estimate_Hx_cond(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    """ Estimate the entropy of a feature X given normal distributions
    collected from K concepts.
    """
    counts_y_given_x, concept_counts = bin_concept_counts(
        initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins)
    # print(counts_y_given_x)
    entropy = 0.0
    dis_sums = 0.0
    conditionals = []
    for y_class in range(len(post_split_mus)):
        conditionals.append([0, 0])
    for bin_class_counts in counts_y_given_x:
        count = np.sum(bin_class_counts)
        if count > 0.0:
            entropy -= count * np.log2(count)
            dis_sums += count
        for y_class, c in enumerate(bin_class_counts):
            if c > 0:
                conditionals[y_class][0] -= c * np.log2(c)
                conditionals[y_class][1] += c

    initial_entropy = (entropy + (dis_sums * np.log2(dis_sums))
                       ) / dis_sums if dis_sums > 0.0 else 0.0
    conditional_entropies = [
        (e + (s * np.log2(s))) / s if s > 0.0 else 0.0 for e, s in conditionals]
    # print(conditional_entropies)
    # print(post_split_sigmas)
    # print([normal_distribution_entropy(0, s) for s in post_split_sigmas])
    # print(initial_entropy)
    return initial_entropy, conditional_entropies


cdf_cache = {}


def bin_X_old(mu, sigma, start, end, num_bins, total_count):
    """ Create bins for a given gaussian
    """
    cumulative_count = 0
    current_value = start
    bin_width = (end-start) / num_bins
    bins = []
    bin_positions = [current_value]
    cdf_count = 0
    for i in range(1, num_bins+1):
        last_value = current_value
        current_value += bin_width
        # To speed up search, don't bother if we are too far
        # below or above mu.
        # Idea is if both bin edges are more than 3xsigma from the mean,
        # which accounts for approx 0.3% of the samples,
        # they will not contain a meaningful amount of samples.
        # Single cdf is relatively expensive, we just count as 0.
        lower_bound = mu - 5*sigma
        upper_bound = mu + 5*sigma
        if (last_value < lower_bound and current_value < lower_bound) or (last_value > upper_bound and current_value > upper_bound):
            cdf_count = cdf_count
        else:
            # cdf_count = int(scipy.stats.norm.cdf(current_value, loc=mu, scale=sigma) * total_count)
            cdf_count = scipy.stats.norm.cdf(
                current_value, loc=mu, scale=sigma) * total_count
        bin_count = cdf_count - cumulative_count
        bins.append(bin_count)
        cumulative_count = cdf_count
        bin_positions.append(current_value)
    return bins, bin_positions


def bin_X(mu, sigma, start, end, num_bins, total_count):
    """ Create bins for a given gaussian
    """
    cumulative_count = 0
    current_value = start
    bin_width = (end-start) / num_bins
    bins = []
    bin_deltas = []
    bin_positions = [current_value]
    cdf_count = 0
    b_d_s = None
    b_d_e = None
    for i in range(1, num_bins+1):
        last_value = current_value
        current_value += bin_width
        # To speed up search, don't bother if we are too far
        # below or above mu.
        # Idea is if both bin edges are more than 3xsigma from the mean,
        # which accounts for approx 0.3% of the samples,
        # they will not contain a meaningful amount of samples.
        # Single cdf is relatively expensive, we just count as 0.
        lower_bound = mu - 5*sigma
        upper_bound = mu + 5*sigma
        if (last_value < lower_bound and current_value < lower_bound) or (last_value > upper_bound and current_value > upper_bound):
            pass
        else:
            if b_d_s is None:
                b_d_s = i - 1
            # cdf_count = int(scipy.stats.norm.cdf(current_value, loc=mu, scale=sigma) * total_count)
            val = (current_value - mu) / sigma
            bin_deltas.append(val)
        bin_positions.append(current_value)

    cdf_counts = scipy.stats.norm.cdf(bin_deltas) * total_count
    bins = np.zeros(num_bins)
    if len(cdf_counts) > 0:
        bin_counts = cdf_counts[1:] - cdf_counts[:-1]
        bins[b_d_s] = cdf_counts[0]
        bins[b_d_s+1:b_d_s+1+len(bin_counts)] = bin_counts
    # t_bins, t_bps = bin_X_old(mu, sigma, start, end, num_bins, total_count)
    # assert((t_bins == bins).all())
    # assert((t_bps == bin_positions))
    return bins, bin_positions


def bin_concept_counts(initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    """ Splits the range into N bins.
    For each of K concept assumed to be distributed N(mu, sigma^2), gives the count of samples
    appearing in each bin.
    Return format:
    [
        [C_11, C_12, ..., C_1K],
        ...
        [C_N1, C_N2, ..., C_NK]
    ]
    Where C_nk is the number of samples seen from concept k appearing in bin n.
    Also returns the total number of samples in each concept.
    """
    concept_counts = []
    counts_y_given_x = [[] for b in range(num_bins)]
    for concept_mu, concept_sigma, concept_count in zip(post_split_mus, post_split_sigmas, post_split_counts):
        concept_weight = 0
        binned_counts, bin_locs = bin_X(concept_mu, max(
            concept_sigma, 0.01), initial_range[0], initial_range[1], num_bins, concept_count)
        for bin_i, bin_count in enumerate(binned_counts):
            counts_y_given_x[bin_i].append(bin_count)
            concept_weight += bin_count
        concept_counts.append(concept_weight)

    return counts_y_given_x, concept_counts


def MI_estimation(initial_mu, initial_sigma, initial_range,  post_split_mus, post_split_sigmas, post_split_counts, num_bins=10):
    """ Constant space approximation of the distribution of Y and Y|X as normal distributions.
    We approximate entropy using bins calculated from normal distributions.
    We estimate bin counts, so use the actual numbers in the bins (concept_counts)
    rather than the real number seen, post_split_counts.
    """
    # np.seterr('raise')
    # initial_entropy = estimate_Hy(post_split_counts)
    binned_Y_counts, concept_counts = bin_concept_counts(
        initial_range, post_split_mus, post_split_sigmas, post_split_counts, num_bins)
    initial_entropy = estimate_Hy(concept_counts)
    total_weight = 0.0
    conditional_entropy = 0.0
    for bin_counts in binned_Y_counts:
        bin_weight = np.sum(bin_counts)
        if bin_weight > 0.0:
            total_weight += bin_weight
            conditional_entropy += bin_weight * estimate_Hy(bin_counts)
    entropy = initial_entropy - \
        (conditional_entropy / total_weight) if total_weight > 0 else 0.0
    np.seterr('warn')
    return entropy


def get_bins_from_fingerprints(concept_fingerprints, source, feature):
    """ Get the counts of each concept seen for each binned value of x.
    Returns:
        concept_counts: total number of each concept across all bins: sum of cols
        concept_counts_given_x: matrix with row per binned value of x, where each col is the observations falling in that bin from a concept.
            Each element bc represents the number of times concept c has x value b.
        bin weights: total number of all concepts in each bin: sum of rows
    """
    num_bins = concept_fingerprints[0].num_bins
    range_start = min([f.normalizer.data_ranges[source][feature][0]
                      for f in concept_fingerprints])
    range_end = max([f.normalizer.data_ranges[source][feature][1]
                    for f in concept_fingerprints])
    concept_counts_given_x = []
    total_weight = 0
    bin_weights = np.zeros(num_bins)
    concept_counts = np.zeros(len(concept_fingerprints))
    for c_i, concept in enumerate(concept_fingerprints):
        observations_per_x = np.empty(num_bins)
        bin_counts, bin_locs = concept.get_binned_feature(
            source, feature, [range_start, range_end])
        for bin_i, bin_count in enumerate(bin_counts):
            observations_per_x[bin_i] = bin_count
            bin_weights[bin_i] += bin_count
            concept_counts[c_i] += bin_count
            total_weight += bin_count
        concept_counts_given_x.append(
            observations_per_x.reshape((num_bins, 1)))
    concept_counts_given_x = np.hstack(concept_counts_given_x)
    return concept_counts, concept_counts_given_x, bin_weights, total_weight


def get_bins_from_fingerprints_fast(concept_fingerprints, source, feature):
    """ Get the counts of each concept seen for each binned value of x.
    Returns:
        concept_counts: total number of each concept across all bins: sum of cols
        concept_counts_given_x: matrix with row per binned value of x, where each col is the observations falling in that bin from a concept.
            Each element bc represents the number of times concept c has x value b.
        bin weights: total number of all concepts in each bin: sum of rows
    """
    num_bins = concept_fingerprints[0].num_bins
    range_start = min([f.normalizer.data_ranges[source][feature][0]
                      for f in concept_fingerprints])
    range_end = max([f.normalizer.data_ranges[source][feature][1]
                    for f in concept_fingerprints])
    concept_counts_given_x = []
    total_weight = 0
    bin_weights = np.zeros(num_bins)
    concept_counts = np.zeros(len(concept_fingerprints))
    for c_i, concept in enumerate(concept_fingerprints):
        bin_counts, bin_locs = concept.get_binned_feature(
            source, feature, [range_start, range_end])
        total_sum = bin_counts.sum()
        bin_weights += bin_counts
        concept_counts[c_i] = total_sum
        total_weight += total_sum
        concept_counts_given_x.append(bin_counts.reshape((num_bins, 1)))
    concept_counts_given_x = np.hstack(concept_counts_given_x)
    # concept_counts_o, concept_counts_given_x_o, bin_weights_o, total_weight_o = get_bins_from_fingerprints(concept_fingerprints, source, feature)
    # assert(np.isclose(concept_counts_o, concept_counts).all())
    # assert(np.isclose(concept_counts_given_x_o, concept_counts_given_x).all())
    # assert(np.isclose(bin_weights_o, bin_weights).all())
    # assert((total_weight_o == total_weight))
    return concept_counts, concept_counts_given_x, bin_weights, total_weight


def MI_estimation_cache(concept_counts, binned_Y_counts, bin_weights, total_weight):
    """ Constant space approximation of the distribution of Y and Y|X as normal distributions.
    We approximate entropy using bins calculated from normal distributions.
    """
    # np.seterr('raise')
    initial_entropy_np = estimate_Hy_np(concept_counts)
    total_weight = 0.0
    conditional_entropy_np = 0.0
    for b_i, bin_counts in enumerate(binned_Y_counts):
        bin_weight = bin_weights[b_i]
        if bin_weight > 0.0:
            total_weight += bin_weight
            conditional_entropy_np += bin_weight * estimate_Hy_np(bin_counts)
    entropy_np = initial_entropy_np - \
        (conditional_entropy_np / total_weight) if total_weight > 0 else 0.0
    np.seterr('warn')
    return entropy_np


def MI_estimation_cache_mat(concept_counts, binned_Y_counts, bin_weights, total_weight):
    """ Constant space approximation of the distribution of Y and Y|X as normal distributions.
    We approximate entropy using bins calculated from normal distributions.
    """
    # np.seterr('raise')
    initial_entropy_np = estimate_Hy_np(concept_counts)
    conditional_entropy_np = 0.0
    np.seterr(all='ignore')
    inverse_bin_weights = np.reciprocal(bin_weights)
    inverse_bin_weights = np.nan_to_num(inverse_bin_weights, posinf=0.0)
    bin_weight_col = inverse_bin_weights.reshape((len(inverse_bin_weights), 1))
    bin_weight_mat = np.hstack(
        [bin_weight_col for i in range(len(concept_counts))])
    bin_concept_probabilities = np.multiply(binned_Y_counts, bin_weight_mat)
    log_probabilities = np.log2(bin_concept_probabilities)
    # np.seterr(all='raise')
    filled_log_probabilities = np.nan_to_num(log_probabilities, neginf=0.0)
    entropy_elements = np.multiply(
        bin_concept_probabilities, filled_log_probabilities)
    bin_entropies = entropy_elements.sum(axis=1) * -1
    weighted_bin_entropies = np.multiply(bin_entropies, bin_weights)
    conditional_entropy_np = np.sum(weighted_bin_entropies)
    # for b_i, bin_counts in enumerate(binned_Y_counts):
    #     bin_weight = bin_weights[b_i]
    #     if bin_weight > 0.0:
    #         total_weight += bin_weight
    #         conditional_entropy_np += bin_weight * estimate_Hy_np(bin_counts)
    entropy_np = initial_entropy_np - \
        (conditional_entropy_np / total_weight) if total_weight > 0 else 0.0
    np.seterr('warn')
    return entropy_np


def histogram_entropy_nonnp(histogram, bins, hist_total):
    """ Calculate histogram entropy.
    Note: Wikipedia formula is weird! Has pk as prob * bin_width
    """
    entropy = 0.0
    if bins[0] == bins[-1]:
        return entropy
    for bin_count in histogram:
        bin_probability = bin_count / hist_total
        # if bin_probability <= 0.0:
        # bin_probability = 1 / hist_total
        # entropy -= bin_probability * np.log2(bin_probability / (bin_end - bin_start))
        if bin_probability > 0.0:
            # entropy -= (bin_probability * (bin_end - bin_start)) * np.log2(bin_probability)
            entropy -= (bin_probability) * np.log2(bin_probability)
    return entropy


def histogram_entropy(histogram, bins, hist_total):
    """ Calculate histogram entropy.
    Note: Wikipedia formula is weird! Has pk as prob * bin_width
    """
    entropy = 0.0
    if bins[0] == bins[-1]:
        return entropy

    bin_probability = histogram / hist_total
    bin_probability = np.nan_to_num(bin_probability, posinf=0.0)
    log_probability = np.log2(bin_probability)
    entropy_elements = np.multiply(bin_probability, log_probability) * -1
    entropy = np.nansum(entropy_elements)
    # for bin_count in histogram:
    #     bin_probability = bin_count / hist_total
    #     # if bin_probability <= 0.0:
    #         # bin_probability = 1 / hist_total
    #     # entropy -= bin_probability * np.log2(bin_probability / (bin_end - bin_start))
    #     if bin_probability > 0.0:
    #         # entropy -= (bin_probability * (bin_end - bin_start)) * np.log2(bin_probability)
    #         entropy -= (bin_probability) * np.log2(bin_probability)
    return entropy

# def histogram_entropy_np(histogram, bins, hist_total):
#     entropy = 0.0
#     bin_width = bins[1] - bins[0]
#     for bin_i, bin_count in enumerate(histogram):
#         if bin_width <= 0:
#             continue
#         bin_probability = bin_count / hist_total
#         if bin_probability <= 0.0:
#             bin_probability = 1 / hist_total
#         entropy -= bin_probability * np.log2(bin_probability / bin_width)
#     return entropy


@jit
def histogram_entropy_jit(histogram, bins, hist_total):
    entropy = 0.0
    for bin_count, (bin_start, bin_end) in zip(histogram, bins):
        if bin_start == bin_end:
            continue
        bin_probability = bin_count / hist_total
        if bin_probability <= 0.0:
            bin_probability = 1 / hist_total
        entropy -= bin_probability * \
            np.log2(bin_probability / (bin_end - bin_start))
    return entropy


def get_bin(value, bins):
    """ Bin value in O(1) given range.
    Assumes bins equally partition range.
    """
    bins_start = bins[0][0]
    # bins_end = bins[-1][1]
    bin_width = bins[0][1] - bins[0][0]
    if bin_width == 0:
        return len(bins) - 1
    dist_from_start = value - bins_start
    bin_i = int(dist_from_start / bin_width)
    bin_i = min(bin_i, len(bins) - 1)
    return bin_i


def get_bin_slow(value, bins):
    """ Bin value in O(N), but can handle
    and bins. 
    Assumes no values less than the first bin_start are
    possible.
    """
    for bin_i, (bin_start, bin_end) in enumerate(bins):
        if bin_start <= value < bin_end:
            return bin_i
    return len(bins) - 1


def get_bin_np(value, bins):
    return np.digitize([value], bins)[0]


@jit
def get_bin_jit(value, bins):
    for bin_i, (bin_start, bin_end) in enumerate(bins):
        if bin_start <= value < bin_end:
            return bin_i
    return len(bins) - 1


def bin_values(values, np_bins):
    new_histogram_np, new_np_bins = np.histogram(values, bins=np_bins)
    return new_histogram_np


def merge_histograms(histograms, bins):
    values = {}
    min_val = sys.maxsize
    max_val = -(sys.maxsize - 1)
    num_bins = len(bins[0])
    total_count = 0
    for histogram, binning in zip(histograms, bins):
        for count, (bin_start, bin_end) in zip(histogram, binning):
            val = (bin_start + bin_end) / 2
            if val not in values:
                values[val] = 0
            values[val] += count
            total_count += count
            min_val = min(val, min_val)
            max_val = max(val, max_val)
    new_bins = []
    # max_val and min_val refer to the middle of the first
    # and last bins. If we fit num_bins - 1 we can find right width
    # to correct.
    inner_width = max_val - min_val
    bin_width = inner_width / (num_bins - 1)
    outer_min = min_val - (bin_width / 2)
    outer_max = max_val + (bin_width / 2)
    total_width = outer_max - outer_max
    for b_i in range(num_bins):
        i = outer_min + b_i * bin_width
        new_bins.append((i, i+bin_width))

    new_histogram = [0 for i in range(num_bins)]
    for val in values:
        new_histogram[get_bin(val, new_bins)] += values[val]
    return new_histogram, new_bins, total_count


def rebin_histogram(histogram, original_bins, new_bins):
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
        new_histogram[get_bin(val, new_bins)] += values[val]
    return new_histogram


def merge_histograms_np(histograms, bins):
    values = {}
    min_val = None
    max_val = None
    num_bins = len(bins[0]) + 1
    total_count = 0
    for histogram, binning in zip(histograms, bins):
        bin_width = binning[1] - binning[0]
        binning_start = binning[0] - bin_width
        for bin_index, bin_count in enumerate(histogram):
            bin_middle = binning_start + (bin_width * (bin_index + 0.5))
            val = bin_middle
            values[val] = values.get(val, 0) + bin_count
            total_count += bin_count
            if min_val is None or val < min_val:
                min_val = val
            if max_val is None or val > max_val:
                max_val = val
    total_width = max_val - min_val
    bin_width = total_width / num_bins
    new_bins = np.zeros(num_bins - 1, dtype=float)
    for b_i in range(0, num_bins - 1):
        i = min_val + (b_i + 1) * bin_width
        new_bins[b_i] = i

    new_histogram = np.zeros(num_bins, dtype=float)
    for val, count in values.items():
        bin_index = get_bin_np(val, new_bins)
        new_histogram[bin_index] += count
    return new_histogram, new_bins, total_count


def merge_histograms_np2(histograms, bins):
    """ We adjust range to keep bin edges
    the same if they don't change. Naively,
    since we use the middle bin val, np.histogram
    shrinks the range. This makes cached hists not usable.
    """
    values = None
    num_bins = len(bins[0]) - 1
    for histogram, binning in zip(histograms, bins):
        bin_values = (binning[:-1] + binning[1:]) / 2.
        hist_values = np.repeat(bin_values, histogram)
        if values is None:
            values = hist_values
        else:
            values = np.concatenate((values, hist_values))
    min_val = values.min()
    max_val = values.max()
    inner_width = max_val - min_val
    bin_width = inner_width / (num_bins - 1)
    outer_min = min_val - (bin_width / 2)
    outer_max = max_val + (bin_width / 2)
    total_width = outer_max - outer_max
    new_histogram, new_bins = np.histogram(
        values, bins=num_bins, range=(outer_min, outer_max))
    return new_histogram, new_bins


@jit
def merge_histograms_jit(histograms, bins):
    min_val = 0
    min_val_set = False
    max_val = 0
    max_val_set = False
    num_bins = len(bins[0])
    total_count = 0
    # for histogram, binning in zip(histograms, bins):
    for i in range(len(histograms)):
        histogram = histograms[i]
        binning = bins[i]
        for b in range(num_bins):
            count = histogram[b]
            bin_start = binning[b][0]
            bin_end = binning[b][1]
        # for count, (bin_start, bin_end) in zip(histogram, binning):
            val = (bin_start + bin_end) / 2
            total_count += count
            if not min_val_set or val < min_val:
                min_val = val
                min_val_set = True
            if not max_val_set or val > max_val:
                max_val = val
                max_val_set = True
    new_bins = List()
    total_width = max_val - min_val
    bin_width = total_width / num_bins
    for b_i in range(num_bins):
        i = min_val + b_i * bin_width
        new_bins.append((i, i+bin_width))

    new_histogram = List()
    for i in range(num_bins):
        new_histogram.append(0)
    for histogram, binning in zip(histograms, bins):
        for count, (bin_start, bin_end) in zip(histogram, binning):
            val = (bin_start + bin_end) / 2
            new_histogram[get_bin_jit(val, new_bins)] += count
    return new_histogram, new_bins, total_count


def MI_histogram_estimation(overall_histogram, overall_bins, overall_count, concept_histograms, concept_bins, concept_counts):
    """ fingerprints in normalizer (overall) and each concept
    store a histogram of MI feature values. 
    We calculate entropy from each histogram.
    """
    # np.seterr('raise')
    # overall_h_x = histogram_entropy(overall_histogram, overall_bins, overall_count)
    merged_histogram, merged_bins, merged_count = merge_histograms(
        concept_histograms, concept_bins)
    h_x = histogram_entropy(merged_histogram, merged_bins, merged_count)
    h_x_given_concept = 0.0
    total_weight = 0.0
    for c_histogram, c_bin, c_weight in zip(concept_histograms, concept_bins, concept_counts):
        total_weight += c_weight
        range_corrected_histogram = rebin_histogram(
            c_histogram, c_bin, merged_bins)
        # concept_entropy = histogram_entropy(c_histogram, c_bin, c_weight) * c_weight
        concept_entropy = histogram_entropy(
            range_corrected_histogram, merged_bins, c_weight) * c_weight
        h_x_given_concept += concept_entropy
    h_x_given_concept /= total_weight

    return max(h_x - h_x_given_concept, 0)


def MI_histogram_estimation_old(overall_histogram, overall_bins, overall_count, concept_histograms, concept_bins, concept_counts):
    """ fingerprints in normalizer (overall) and each concept
    store a histogram of MI feature values. 
    We calculate entropy from each histogram.
    """
    # np.seterr('raise')
    # overall_h_x = histogram_entropy(overall_histogram, overall_bins, overall_count)
    merged_histogram, merged_bins, merged_count = merge_histograms(
        concept_histograms, concept_bins)
    h_x = histogram_entropy(merged_histogram, merged_bins, merged_count)
    h_x_given_concept = 0.0
    total_weight = 0.0
    for c_histogram, c_bin, c_weight in zip(concept_histograms, concept_bins, concept_counts):
        total_weight += c_weight
        # range_corrected_histogram = rebin_histogram(c_histogram, c_bin, merged_bins)
        concept_entropy = histogram_entropy(
            c_histogram, c_bin, c_weight) * c_weight
        # concept_entropy = histogram_entropy(range_corrected_histogram, merged_bins, c_weight) * c_weight
        h_x_given_concept += concept_entropy
    h_x_given_concept /= total_weight

    return max(h_x - h_x_given_concept, 0)

# def MI_histogram_estimation_cache(overall_histogram_x, overall_bins_x, overall_count, concept_histograms_x, concept_bins_x, concept_counts_x):
#     """ fingerprints in normalizer (overall) and each concept
#     store a histogram of MI feature values.
#     We calculate entropy from each histogram.
#     """
#     np.seterr('raise')
#     overall_histogram_y = concept_histograms_x.sum(axis=1).reshape(1,)
#     y_bins = np.array(range(0, concept_histograms_x.shape[0]))
#     concept_histograms_y = concept_histograms_x.transpose()
#     concept_counts_y = concept_histograms_y.sum(axis=1)
#     h_x = histogram_entropy(overall_histogram, y_bins, overall_count)
#     h_x_given_concept = 0.0
#     total_weight = 0.0
#     for c_histogram, c_weight in zip(concept_histograms_y, concept_counts_y):
#         total_weight += c_weight
#         # range_corrected_histogram = rebin_histogram(c_histogram, c_bin, merged_bins)
#         concept_entropy = histogram_entropy(c_histogram, y_bins, c_weight) * c_weight
#         # concept_entropy = histogram_entropy(range_corrected_histogram, merged_bins, c_weight) * c_weight
#         h_x_given_concept += concept_entropy
#     h_x_given_concept /= total_weight

    return max(h_x - h_x_given_concept, 0)


def MI_histogram_estimation_cache_mat(overall_histogram, overall_bins, overall_count, concept_histograms, concept_bins, concept_counts, total_weight=None):
    """ fingerprints in normalizer (overall) and each concept
    store a histogram of MI feature values. 
    We calculate entropy from each histogram.
    """
    np.seterr('ignore')
    # TODO: <testing>
    # overall_histogram_y = concept_histograms.sum(axis=1)
    # concept_bins_y = np.array(range(0, concept_histograms.shape[0]))
    # concept_histograms_y = concept_histograms.transpose()
    # concept_counts_y = concept_histograms_y.sum(axis=1)
    # h_x_by = histogram_entropy(overall_histogram_y, concept_bins_y, overall_count)
    # IG_byy = MI_histogram_estimation_cache_mat_byy(overall_histogram, overall_bins, overall_count, concept_histograms, concept_bins, concept_counts, total_weight = None)
    # np.seterr('ignore')
    # </testing>
    h_x = histogram_entropy(overall_histogram, overall_bins, overall_count)
    if total_weight is None:
        total_weight = np.sum(concept_counts)
    count_matrix = concept_histograms
    inverse_row_weight = np.reciprocal(concept_counts, dtype=float)
    inverse_row_weight = np.nan_to_num(inverse_row_weight, posinf=0.0)
    probability_matrix = count_matrix * inverse_row_weight[:, np.newaxis]
    log_probabilities = np.log2(probability_matrix)
    filled_log_probabilities = log_probabilities
    entropy_elements = np.multiply(
        probability_matrix, filled_log_probabilities)
    concept_entropies = np.nansum(entropy_elements, axis=1) * -1
    weighted_bin_entropies = np.multiply(concept_entropies, concept_counts)
    conditional_entropy_sum = np.nansum(weighted_bin_entropies)
    h_x_given_concept = conditional_entropy_sum / total_weight
    # np.seterr('raise')

    return max(h_x - h_x_given_concept, 0)


def MI_histogram_estimation_cache_mat_byy(overall_histogram_x, overall_bins_x, overall_count, concept_histograms_x, concept_bins_x, concept_counts_x, total_weight=None):
    """ fingerprints in normalizer (overall) and each concept
    store a histogram of MI feature values. 
    We calculate entropy from each histogram.
    """
    np.seterr('ignore')
    overall_histogram = concept_histograms_x.sum(axis=1)
    concept_bins = np.array(range(0, concept_histograms_x.shape[0]))
    concept_histograms = concept_histograms_x.transpose()
    concept_counts = concept_histograms.sum(axis=1)
    h_x = histogram_entropy(overall_histogram, concept_bins, overall_count)
    if total_weight is None:
        total_weight = np.sum(concept_counts)
    count_matrix = concept_histograms
    inverse_row_weight = np.reciprocal(concept_counts, dtype=float)
    inverse_row_weight = np.nan_to_num(inverse_row_weight, posinf=0.0)
    probability_matrix = count_matrix * inverse_row_weight[:, np.newaxis]
    log_probabilities = np.log2(probability_matrix)
    filled_log_probabilities = log_probabilities
    entropy_elements = np.multiply(
        probability_matrix, filled_log_probabilities)
    concept_entropies = np.nansum(entropy_elements, axis=1) * -1
    weighted_bin_entropies = np.multiply(concept_entropies, concept_counts)
    conditional_entropy_sum = np.nansum(weighted_bin_entropies)
    h_x_given_concept = conditional_entropy_sum / total_weight
    # np.seterr('raise')

    return max(h_x - h_x_given_concept, 0)


def MI_histogram_estimation_np(overall_histogram, overall_bins, overall_count, concept_histograms, concept_bins, concept_counts):
    """ fingerprints in normalizer (overall) and each concept
    store a histogram of MI feature values. 
    We calculate entropy from each histogram.
    """
    # np.seterr('raise')
    # overall_h_x = histogram_entropy(overall_histogram, overall_bins, overall_count)
    merged_histogram, merged_bins, merged_count = merge_histograms_np(
        concept_histograms, concept_bins)
    h_x = histogram_entropy_np(merged_histogram, merged_bins, merged_count)
    h_x_given_concept = 0.0
    total_weight = 0.0
    for c_histogram, c_bin, c_weight in zip(concept_histograms, concept_bins, concept_counts):
        total_weight += c_weight
        concept_entropy = histogram_entropy_np(
            c_histogram, c_bin, c_weight) * c_weight
        h_x_given_concept += concept_entropy
    h_x_given_concept /= total_weight

    return max(h_x - h_x_given_concept, 0)


@jit
def MI_histogram_estimation_jit(overall_histogram, overall_bins, overall_count, concept_histograms, concept_bins, concept_counts):
    """ fingerprints in normalizer (overall) and each concept
    store a histogram of MI feature values. 
    We calculate entropy from each histogram.
    """
    # np.seterr('raise')
    # overall_h_x = histogram_entropy(overall_histogram, overall_bins, overall_count)
    merged_histogram, merged_bins, merged_count = merge_histograms_jit(
        concept_histograms, concept_bins)
    h_x = histogram_entropy_jit(merged_histogram, merged_bins, merged_count)
    h_x_given_concept = 0.0
    total_weight = 0.0
    for c_histogram, c_bin, c_weight in zip(concept_histograms, concept_bins, concept_counts):
        total_weight += c_weight
        concept_entropy = histogram_entropy_jit(
            c_histogram, c_bin, c_weight) * c_weight
        h_x_given_concept += concept_entropy
    h_x_given_concept /= total_weight

    return max(h_x - h_x_given_concept, 0)


def MI_histogram_estimation_jl(overall_histogram, overall_bins, overall_count, concept_histograms, concept_bins, concept_counts):

    Main.overall_histogram = overall_histogram
    Main.overall_bins = overall_bins
    Main.overall_count = overall_count
    Main.concept_histograms = concept_histograms
    Main.concept_bins = concept_bins
    Main.concept_counts = concept_counts
    return jl.eval("MI_histogram_estimation(overall_histogram, overall_bins, overall_count, concept_histograms, concept_bins, concept_counts)")
