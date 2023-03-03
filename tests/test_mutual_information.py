from ConceptFingerprint.Classifier.feature_selection.mutual_information import normal_distribution_entropy, information_gain_normal_distributions, information_gain_normal_distributions_KL, information_gain_normal_distributions_swap, information_gain_normal_distributions_sym, KL_divergence
from ConceptFingerprint.Classifier.feature_selection.mutual_information import information_gain_normal_distributions_Hx, information_gain_normal_distributions_HxCond, bin_X, bin_concept_counts, MI_estimation
from ConceptFingerprint.Classifier.feature_selection.mutual_information import MI_histogram_estimation_np, MI_histogram_estimation_jit, MI_histogram_estimation_jl, information_gain_normal_distributions_UN, histogram_entropy, MI_histogram_estimation
import numpy as np
import pandas as pd
import math
import random
from sklearn.feature_selection import mutual_info_classif
from ConceptFingerprint.Classifier.feature_selection.online_feature_selection import (
    mi_from_fingerprint_histogram_cache
)
from ConceptFingerprint.Classifier.fingerprint import (FingerprintBinningCache, FingerprintBinningCache)
from ConceptFingerprint.Classifier.normalizer import Normalizer
import numpy as np
import time
import pytest


def test_normal_distribution_entropy():
    test_mu = 0
    test_sigma = 1
    ent = normal_distribution_entropy(test_mu, test_sigma)
    assert math.isclose(ent, 2.05, rel_tol=0.01)
    test_sigma = 0.5
    ent = normal_distribution_entropy(test_mu, test_sigma)
    assert math.isclose(ent, 1.05, rel_tol=0.01)

def test_information_gain_normal_distributions():
    dist_1 = np.random.normal(1.75, 0.5, 100000)
    y_1 = np.array([0 for i in range(100000)])
    dist_2 = np.random.normal(0, 0.5, 100000)
    y_2 = np.array([1 for i in range(100000)])

    all_X = np.concatenate((dist_1, dist_2))
    all_y = np.concatenate((y_1, y_2))

    # Assuming a normal distribution, what is the reduction in mutual information between X and y?
    information_gain = information_gain_normal_distributions_UN(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in [dist_1, dist_2]], [np.std(x) for x in [dist_1, dist_2]], [100, 100])
    information_gain_2 = information_gain_normal_distributions_KL(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in [dist_1, dist_2]], [np.std(x) for x in [dist_1, dist_2]], [100, 100])

    print(mutual_info_classif(all_X.reshape(-1, 1), all_y))
    print(information_gain_2)
    print(information_gain)
    # from example at: https://bumps.readthedocs.io/en/latest/guide/entropy.html
    # info gain should be close to 1.
    # sigma of the whole distribution is ~1
    assert math.isclose(np.std(all_X), 1, rel_tol=0.1)
    # average sigma considering y is ~0.5
    assert math.isclose(np.mean((np.std(dist_1), np.std(dist_2))), 0.5, rel_tol=0.1)
    # Entropy of the original is ~2.05
    print(normal_distribution_entropy(0, np.std(all_X)))
    assert math.isclose(normal_distribution_entropy(0, np.std(all_X)), 2.05, rel_tol=0.1)

    # Average new entropy is ~1.05
    assert math.isclose(np.mean((normal_distribution_entropy(0, np.std(dist_1)), normal_distribution_entropy(0, np.std(dist_2)))), 1.05, rel_tol=0.2)
    # So info gain is 2.05 - 1.05 = ~1
    assert math.isclose(information_gain, 1, rel_tol=0.2)


def plot_information_gain_normal_distributions():
    observations = []
    for i in range(100):
        mu_A = random.random()
        sigma_A = random.random()
        mu_B = random.random()
        sigma_B = random.random()
        dist_1 = np.random.normal(mu_A, sigma_A, 100000)
        y_1 = np.array([0 for i in range(100000)])
        dist_2 = np.random.normal(mu_B, sigma_B, 100000)
        y_2 = np.array([1 for i in range(100000)])

        all_X = np.concatenate((dist_1, dist_2))
        all_y = np.concatenate((y_1, y_2))

        # Assuming a normal distribution, what is the reduction in mutual information between X and y?
        information_gain = information_gain_normal_distributions_UN(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in [dist_1, dist_2]], [np.std(x) for x in [dist_1, dist_2]], [100, 100])
        information_gain_2 = information_gain_normal_distributions_KL(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in [dist_1, dist_2]], [np.std(x) for x in [dist_1, dist_2]], [100, 100])
        information_gain_swap = information_gain_normal_distributions_swap(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in [dist_1, dist_2]], [np.std(x) for x in [dist_1, dist_2]], [100, 100])
        information_gain_sym = information_gain_normal_distributions_sym(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in [dist_1, dist_2]], [np.std(x) for x in [dist_1, dist_2]], [100, 100])

        print(mutual_info_classif(all_X.reshape(-1, 1), all_y))
        print(information_gain)
        print(information_gain_2)
        print(information_gain_swap)
        print(information_gain_sym)

        observation = {
            'mu_A': mu_A,
            'sigma_A': sigma_A,
            'mu_B': mu_B,
            'sigma_B': sigma_B,
            'mu_O': np.mean(all_X),
            'sigma_O': np.std(all_X),
            'information_gain': information_gain,
            'information_gain_2': information_gain_2,
            'information_gain_swap': information_gain_swap,
            'information_gain_sym': information_gain_sym,
        }
        observations.append(observation)
    
    df = pd.DataFrame(observations)
    print(df.head())

def test_bin_X():
    mu=1
    sigma=0.5
    data = np.random.normal(loc=mu, scale=sigma, size=1000)

    num_bins = 10
    data_mu = np.mean(data)
    data_sigma = np.std(data)
    numpy_bin_counts, numpy_bin_locs = np.histogram(data)
    my_bin_counts, my_bin_locs = bin_X(data_mu, data_sigma, np.min(data), np.max(data), num_bins=num_bins, total_count=len(data))
    # Bins should be in the same place
    for my_loc, np_loc in zip(my_bin_locs, numpy_bin_locs):
        assert math.isclose(my_loc, np_loc, rel_tol=0.01)
    # Numpy is calculating using the sample, we are estimating
    # with a gaussian. Should be close but not exact.
    for my_count, np_count in zip(my_bin_counts, numpy_bin_counts):
        print(my_count, np_count)
        assert abs(my_count - np_count) < max(np_count * 0.3, 20)

def test_bin_concept_counts():
    mu_C1=np.random.random()
    sigma_C1=np.random.random()
    data_C1 = np.random.normal(loc=mu_C1, scale=sigma_C1, size=1000)
    mu_data_C1 = np.mean(data_C1)
    sigma_data_C1 = np.std(data_C1)
    range_C1 = [np.min(data_C1), np.max(data_C1)]
    mu_C2=np.random.random()
    sigma_C2=np.random.random()
    data_C2 = np.random.normal(loc=mu_C2, scale=sigma_C2, size=1000)
    mu_data_C2 = np.mean(data_C2)
    sigma_data_C2 = np.std(data_C2)
    range_C2 = [np.min(data_C2), np.max(data_C2)]
    mu_C3=np.random.random()
    sigma_C3=np.random.random()
    data_C3 = np.random.normal(loc=mu_C3, scale=sigma_C3, size=1000)
    mu_data_C3 = np.mean(data_C3)
    sigma_data_C3 = np.std(data_C3)
    range_C3 = [np.min(data_C3), np.max(data_C3)]

    data_total = [*data_C1, *data_C2, *data_C3]
    total_mu = np.mean(data_total)
    total_sigma = np.std(data_total)
    total_range = [np.min(data_total), np.max(data_total)]

    C1_bin_counts, _ = bin_X(mu_data_C1, sigma_data_C1, total_range[0], total_range[1], 10, len(data_C1))
    C2_bin_counts, _ = bin_X(mu_data_C2, sigma_data_C2, total_range[0], total_range[1], 10, len(data_C2))
    C3_bin_counts, _ = bin_X(mu_data_C3, sigma_data_C3, total_range[0], total_range[1], 10, len(data_C3))
    print(list(zip(C1_bin_counts, C2_bin_counts, C3_bin_counts)))

    print(bin_concept_counts(total_range, [mu_data_C1, mu_data_C2, mu_data_C3], [sigma_data_C1, sigma_data_C2, sigma_data_C3], [1000, 1000, 1000]))

@pytest.mark.skip(reason="no asserts")
def test_MI_estimation():
    l_1 = math.floor(np.random.random() * 5000)
    dist_1 = np.random.normal(np.random.random() * 10, np.random.random() * 2, l_1)
    y_1 = np.array([0 for i in range(l_1)])
    l_2 = math.floor(np.random.random() * 5000)
    dist_2 = np.random.normal(np.random.random() * 10, np.random.random() * 2, l_2)
    y_2 = np.array([1 for i in range(l_2)])

    all_X = np.concatenate((dist_1, dist_2))
    all_y = np.concatenate((y_1, y_2))

    # Assuming a normal distribution, what is the reduction in mutual information between X and y?
    information_gain = MI_estimation(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in [dist_1, dist_2]], [np.std(x) for x in [dist_1, dist_2]], [len(dist_1), len(dist_2)], 10)

    print(mutual_info_classif(all_X.reshape(-1, 1), all_y))
    print(information_gain)

@pytest.mark.skip(reason="no asserts")
def test_information_gain_normal_distributions_HxCond():
    mu_A = random.random() * 100
    sigma_A = random.random() * 20
    mu_B = random.random() * 100
    sigma_B = random.random() * 20
    mu_C = random.random() * 100
    sigma_C = random.random() * 20
    mu_D = random.random() * 100
    sigma_D = random.random() * 20
    dist_1 = np.random.normal(mu_A, sigma_A, 100000)
    y_1 = np.array([0 for i in range(100000)])
    dist_2 = np.random.normal(mu_B, sigma_B, 100000)
    y_2 = np.array([1 for i in range(100000)])
    dist_3 = np.random.normal(mu_C, sigma_C, 100000)
    y_3 = np.array([2 for i in range(100000)])
    dist_4 = np.random.normal(mu_D, sigma_D, 100000)
    y_4 = np.array([3 for i in range(100000)])

    dists = [dist_1, dist_2, dist_3, dist_4]
    ys = (y_1, y_2, y_3, y_4)
    mus = [mu_A, mu_B, mu_C, mu_D]
    sigmas = [sigma_A, sigma_B, sigma_C, sigma_D]
    dists = [dist_1, dist_2, dist_3]
    ys = (y_1, y_2, y_3)
    mus = [mu_A, mu_B, mu_C]
    sigmas = [sigma_A, sigma_B, sigma_C]
    all_X = np.concatenate(dists)
    all_y = np.concatenate(ys)
    information_gain_bin_HxCond = information_gain_normal_distributions_HxCond(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 1000)
    information_gain = information_gain_normal_distributions(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists])
    skl_IG = mutual_info_classif(all_X.reshape(-1, 1), all_y)[0]
    information_gain_bin_100 = MI_estimation(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 100)
    information_gain_bin_Hx = information_gain_normal_distributions_Hx(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 1000)


    print(skl_IG)
    print(information_gain)
    print(information_gain_bin_100)
    print(information_gain_bin_HxCond)
    print(information_gain_bin_Hx)

@pytest.mark.skip(reason="no asserts")
def test_histogram_entropy():
    norm = Normalizer(fingerprint_constructor = FingerprintBinningCache)
    stats = {
        "test_source1": {"test_feature1": np.random.normal(10, 1), "test_feature2": np.random.rand()},
        "test_source2": {"test_feature1": np.random.rand() * 10, "test_feature2": np.random.normal(10, 5)},
        "test_source3": {"test_feature1": np.random.rand() * 1, "test_feature2": np.random.normal(0, 1)},
    }
    norm.add_stats(stats)
    fp_c1 = FingerprintBinningCache(stats, norm)

    for i in range(2, 100):
        stats = {
            "test_source1": {"test_feature1": np.random.normal(10, 1), "test_feature2": np.random.rand()},
            "test_source2": {"test_feature1": np.random.rand() * 10, "test_feature2": np.random.normal(10, 5)},
            "test_source3": {"test_feature1": np.random.rand() * 1, "test_feature2": np.random.normal(0, 1)},
        }
        norm.add_stats(stats)
        fp_c1.incorperate(stats)
    stats = {
        "test_source1": {"test_feature1": np.random.normal(1, 1), "test_feature2": np.random.rand() * 10},
        "test_source2": {"test_feature1": np.random.rand() * 10, "test_feature2": np.random.normal(10, 5)},
        "test_source3": {"test_feature1": np.random.rand() * 1, "test_feature2": np.random.normal(0, 1)},
    }
    fp_c2 = FingerprintBinningCache(stats, norm)
    for i in range(2, 100):
        stats = {
            "test_source1": {"test_feature1": np.random.normal(1, 1), "test_feature2": np.random.rand() * 10},
            "test_source2": {"test_feature1": np.random.rand() * 10, "test_feature2": np.random.normal(10, 5)},
            "test_source3": {"test_feature1": np.random.rand() * 1, "test_feature2": np.random.normal(0, 1)},
        }
        norm.add_stats(stats)
        fp_c2.incorperate(stats)

    for s in ['test_source1', 'test_source2', 'test_source3']:
        for f in ['test_feature1', 'test_feature2']:
            print(s, f, fp_c1.fingerprint[s][f]['Range'], fp_c1.fingerprint[s][f]['Bins'], fp_c1.fingerprint[s][f]['Histogram'])
            print(s, f, fp_c2.fingerprint[s][f]['Range'], fp_c2.fingerprint[s][f]['Bins'], fp_c2.fingerprint[s][f]['Histogram'])
            print(MI_histogram_estimation(norm.fingerprint.fingerprint[s][f]["Histogram"], norm.fingerprint.fingerprint[s][f]["Bins"], norm.fingerprint.fingerprint[s][f]["seen"], [fi.fingerprint[s][f]["Histogram"] for fi in [fp_c1, fp_c2]], [fi.fingerprint[s][f]["Bins"] for fi in [fp_c1, fp_c2]], [fi.fingerprint[s][f]["seen"] for fi in [fp_c1, fp_c2]]))
            print(mi_from_fingerprint_histogram_cache(norm, [fp_c1, fp_c2], s, f))
            print(histogram_entropy(norm.fingerprint.fingerprint[s][f]["Histogram"], norm.fingerprint.fingerprint[s][f]["Bins"], norm.fingerprint.fingerprint[s][f]["seen"]))
            for fi in [fp_c1, fp_c2]:
                print(histogram_entropy(fi.fingerprint[s][f]["Histogram"], fi.fingerprint[s][f]["Bins"], fi.fingerprint[s][f]["seen"]))

@pytest.mark.skip(reason="no asserts")
def test_histogram_entropy_jl():
    norm = Normalizer(fingerprint_constructor = FingerprintBinningCache)
    stats = {
        "test_source1": {"test_feature1": np.random.normal(10, 1), "test_feature2": np.random.rand()},
        "test_source2": {"test_feature1": np.random.rand() * 10, "test_feature2": np.random.normal(10, 5)},
        "test_source3": {"test_feature1": np.random.rand() * 1, "test_feature2": np.random.normal(0, 1)},
    }
    norm.add_stats(stats)
    fp_c1 = FingerprintBinningCache(stats, norm)

    for i in range(2, 100):
        stats = {
            "test_source1": {"test_feature1": np.random.normal(10, 1), "test_feature2": np.random.rand()},
            "test_source2": {"test_feature1": np.random.rand() * 10, "test_feature2": np.random.normal(10, 5)},
            "test_source3": {"test_feature1": np.random.rand() * 1, "test_feature2": np.random.normal(0, 1)},
        }
        norm.add_stats(stats)
        fp_c1.incorperate(stats)
    stats = {
        "test_source1": {"test_feature1": np.random.normal(1, 1), "test_feature2": np.random.rand() * 10},
        "test_source2": {"test_feature1": np.random.rand() * 10, "test_feature2": np.random.normal(10, 5)},
        "test_source3": {"test_feature1": np.random.rand() * 1, "test_feature2": np.random.normal(0, 1)},
    }
    fp_c2 = FingerprintBinningCache(stats, norm)
    for i in range(2, 100):
        stats = {
            "test_source1": {"test_feature1": np.random.normal(1, 1), "test_feature2": np.random.rand() * 10},
            "test_source2": {"test_feature1": np.random.rand() * 10, "test_feature2": np.random.normal(10, 5)},
            "test_source3": {"test_feature1": np.random.rand() * 1, "test_feature2": np.random.normal(0, 1)},
        }
        norm.add_stats(stats)
        fp_c2.incorperate(stats)

    for s in ['test_source1', 'test_source2', 'test_source3']:
        for f in ['test_feature1', 'test_feature2']:
            print("---------")
            print(s, f)
            print(s, f, fp_c1.fingerprint[s][f]['Range'], fp_c1.fingerprint[s][f]['Bins'], fp_c1.fingerprint[s][f]['Histogram'])
            print(s, f, fp_c2.fingerprint[s][f]['Range'], fp_c2.fingerprint[s][f]['Bins'], fp_c2.fingerprint[s][f]['Histogram'])
            print(MI_histogram_estimation(norm.fingerprint.fingerprint[s][f]["Histogram"], norm.fingerprint.fingerprint[s][f]["Bins"], norm.fingerprint.fingerprint[s][f]["seen"], [fi.fingerprint[s][f]["Histogram"] for fi in [fp_c1, fp_c2]], [fi.fingerprint[s][f]["Bins"] for fi in [fp_c1, fp_c2]], [fi.fingerprint[s][f]["seen"] for fi in [fp_c1, fp_c2]]))
            print(MI_histogram_estimation_jl(norm.fingerprint.fingerprint[s][f]["Histogram"], norm.fingerprint.fingerprint[s][f]["Bins"], norm.fingerprint.fingerprint[s][f]["seen"], [fi.fingerprint[s][f]["Histogram"] for fi in [fp_c1, fp_c2]], [fi.fingerprint[s][f]["Bins"] for fi in [fp_c1, fp_c2]], [fi.fingerprint[s][f]["seen"] for fi in [fp_c1, fp_c2]]))

@pytest.mark.skip(reason="no asserts")
def test_histogram_entropy_timing_jl():
    stats_list = []
    for i in range(0, 25):
        stats = {
            "test_source1": {"test_feature1": np.random.normal(10, 1), "test_feature2": np.random.rand()},
            "test_source2": {"test_feature1": np.random.rand() * 10, "test_feature2": np.random.normal(10, 25)},
            "test_source3": {"test_feature1": np.random.rand() * 1, "test_feature2": np.random.normal(0, 1)},
        }
        stats_list.append(stats)
    for i in range(0, 25):
        stats = {
            "test_source1": {"test_feature1": np.random.normal(1, 1), "test_feature2": np.random.rand() * 10},
            "test_source2": {"test_feature1": np.random.rand() * 10, "test_feature2": np.random.normal(10, 25)},
            "test_source3": {"test_feature1": np.random.rand() * 1, "test_feature2": np.random.normal(0, 1)},
        }
        stats_list.append(stats)
    
    py_start = time.time()
    py_entropy = []
    norm = Normalizer(fingerprint_constructor = FingerprintBinningCache)
    fp_c1 = None
    fp_c2 = None
    for i, stats in enumerate(stats_list):
        norm.add_stats(stats)
        if i < 25:
            if fp_c1 is None:
                fp_c1 = FingerprintBinningCache(stats, norm)
            else:
                fp_c1.incorperate(stats)
        else:
            if fp_c2 is None:
                fp_c2 = FingerprintBinningCache(stats, norm)
            else:
                fp_c2.incorperate(stats)
            for s in ['test_source1', 'test_source2', 'test_source3']:
                for f in ['test_feature1', 'test_feature2']:
                    py_entropy.append(MI_histogram_estimation(norm.fingerprint.fingerprint[s][f]["Histogram"], norm.fingerprint.fingerprint[s][f]["Bins"], norm.fingerprint.fingerprint[s][f]["seen"], [fi.fingerprint[s][f]["Histogram"] for fi in [fp_c1, fp_c2]], [fi.fingerprint[s][f]["Bins"] for fi in [fp_c1, fp_c2]], [fi.fingerprint[s][f]["seen"] for fi in [fp_c1, fp_c2]]))
    py_end = time.time()
            
    jl_start = time.time()
    jl_entropy = []
    norm = Normalizer(fingerprint_constructor = FingerprintBinningCache)
    fp_c1 = None
    fp_c2 = None
    for i, stats in enumerate(stats_list):
        norm.add_stats(stats)
        if i < 5:
            if fp_c1 is None:
                fp_c1 = FingerprintBinningCache(stats, norm)
            else:
                fp_c1.incorperate(stats)
        else:
            if fp_c2 is None:
                fp_c2 = FingerprintBinningCache(stats, norm)
            else:
                fp_c2.incorperate(stats)
            for s in ['test_source1', 'test_source2', 'test_source3']:
                for f in ['test_feature1', 'test_feature2']:
                    jl_entropy.append(MI_histogram_estimation_jl(norm.fingerprint.fingerprint[s][f]["Histogram"], norm.fingerprint.fingerprint[s][f]["Bins"], norm.fingerprint.fingerprint[s][f]["seen"], [fi.fingerprint[s][f]["Histogram"] for fi in [fp_c1, fp_c2]], [fi.fingerprint[s][f]["Bins"] for fi in [fp_c1, fp_c2]], [fi.fingerprint[s][f]["seen"] for fi in [fp_c1, fp_c2]]))
    jl_end = time.time()

    print(f"PY took: {py_end - py_start}")
    print(f"jl took: {jl_end - jl_start}")
    print(f"They are the same: {py_entropy == jl_entropy}")       

# test_histogram_entropy_jl()
# test_histogram_entropy_timing_numpy()
# test_histogram_entropy_eq_numpy()
# test_normal_distribution_entropy()
# test_information_gain_normal_distributions()
# test_information_gain_normal_distributions()
# # plot_information_gain_normal_distributions()
# # print(KL_divergence(0, 1, 0, 1))
# test_bin_X()
# test_bin_concept_counts()
# test_MI_estimation()
# test_information_gain_normal_distributions_HxCond()