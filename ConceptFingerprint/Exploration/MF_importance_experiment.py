import statistics
import pickle
import json
import pathlib
from collections import deque
from itertools import chain, combinations
import multiprocessing as mp
import argparse
import shutil
import math
import os,time

from pyinstrument import Profiler

import tqdm
import scipy.stats
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import cosine
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import acf, pacf
import shap
# from ConceptFingerprint.Exploration.tree import TreeExplainer
from PyEMD import EMD
from entropy import sample_entropy, perm_entropy
from ConceptFingerprint.Data.load_data import \
        load_synthetic_concepts,\
        load_real_concepts,\
        get_inorder_concept_ranges,\
        AbruptDriftStream
from ConceptFingerprint.Exploration.sliding_window_MI import learn_fingerprints_from_concepts, get_sliding_window_similarities
from ConceptFingerprint.Exploration.explore_SW import get_data, run_eval

np.seterr(divide='ignore', invalid='ignore')
class Normalizer:
    def __init__(self):
        self.classes = []
        self.sources = set()
        self.features = set()
        self.data_ranges = {}
        self.seen_classes = 0
        self.seen_stats = 0

    def add_class(self, c):
        if c not in self.classes:
            self.classes.append(c)
            self.classes.sort()
        self.seen_classes += 1

    def add_stats(self, stats):
        self.seen_stats += 1
        for source in stats.keys():
            self.sources.add(source)
            if source not in self.data_ranges:
                self.data_ranges[source] = {}
            for feature in stats[source].keys():
                self.features.add(feature)
                if feature not in self.data_ranges[source]:
                    self.data_ranges[source][feature] = [None, None]

                value = stats[source][feature]
                value_range = self.data_ranges[source][feature]
                value_range[0] = min(value, value_range[0])\
                    if value_range[0] is not None else value
                value_range[1] = max(value, value_range[1])\
                    if value_range[1] is not None else value

    def merge(self, other):
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

    def get_normed_value(self, source, feature, value):
        value_range = self.data_ranges[source][feature]
        width = value_range[1] - value_range[0]
        return (value - value_range[0]) / (width) if width > 0 else value


def train_concept_windows(concepts, normalizer, storage_path, options=None):
    """ Simulate a concept that has been seen before n times.
    A model is trained and a window of observations are taken.
    """
    train_len = options['train_len']
    window_size = options['window_size']

    windows = []
    skipped = False
    for rep in tqdm.tqdm(range(10)):
        for concept_gen, concept_name in tqdm.tqdm(concepts):
            file_store_path = storage_path / f"{concept_name}_{train_len}_{rep}"
            if file_store_path.exists():
                with file_store_path.open('rb') as f:
                    window = pickle.load(f)
                windows.append(window)
                skipped = True
                continue
            windows.append([[], concept_name])

            model = HoeffdingTree()
            for i in range(train_len):
                if concept_gen.n_remaining_samples() == 0:
                    concept_gen.restart()
                X, y = concept_gen.next_sample()
                normalizer.add_class(y[0])
                model.partial_fit(X, y, normalizer.classes)

            for i in range(window_size):
                if concept_gen.n_remaining_samples() == 0:
                    concept_gen.restart()
                X, y = concept_gen.next_sample()
                p = model.predict(X)
                e = p[0] == y[0]
                windows[-1][0].append((X[0], y[0], p[0], e))
            windows[-1].append(model)

            features, labels, predictions, errors, error_distances =\
                window_to_timeseries(windows[-1])
            stats = get_concept_stats([features,
                                       labels,
                                       predictions,
                                       errors,
                                       error_distances], model)
            normalizer.add_stats(stats)
            windows[-1].append(stats)

            with file_store_path.open('wb') as f:
                pickle.dump(windows[-1], f)

    normalizer_path = storage_path / f"{train_len}_normalizer"
    with normalizer_path.open('wb') as f:
        pickle.dump(normalizer, f)

    return windows, skipped

def average_of_concept_stats(concept_stats):
    stats_by_concept = {}
    stats_by_concept_avg = {}
    for stats in concept_stats:        
        for source in stats:
            if source not in stats_by_concept:
                stats_by_concept[source] = {}
            for feature in stats[source]:
                if feature not in stats_by_concept[source]:
                    stats_by_concept[source][feature] = []
                stats_by_concept[source][feature].append(stats[source][feature])

    for source in stats_by_concept:
        if source not in stats_by_concept_avg:
            stats_by_concept_avg[source] = {}
        for feature in stats_by_concept[source]:
            stats_by_concept_avg[source][feature] = np.mean(stats_by_concept[source][feature])
    
    return stats_by_concept_avg


def average_concept_stats(windows):
    stats_by_concept = {}
    stats_by_concept_avg = {}
    for w in windows:
        concept_name = w[1]
        concept_stats = w[3]
        if concept_name not in stats_by_concept:
            stats_by_concept[concept_name] = {}
        
        for source in concept_stats:
            if source not in stats_by_concept[concept_name]:
                stats_by_concept[concept_name][source] = {}
            for feature in concept_stats[source]:
                if feature not in stats_by_concept[concept_name][source]:
                    stats_by_concept[concept_name][source][feature] = []
                stats_by_concept[concept_name][source][feature].append(concept_stats[source][feature])

    for concept_name in stats_by_concept:
        if concept_name not in stats_by_concept_avg:
            stats_by_concept_avg[concept_name] = {}
        for source in stats_by_concept[concept_name]:
            if source not in stats_by_concept_avg[concept_name]:
                stats_by_concept_avg[concept_name][source] = {}
            for feature in stats_by_concept[concept_name][source]:
                stats_by_concept_avg[concept_name][source][feature] = np.mean(stats_by_concept[concept_name][source][feature])
    
    return stats_by_concept_avg

def window_to_timeseries(window):
    features = []
    for f in window[0][0][0]:
        features.append([])
    labels = []
    predictions = []
    errors = []
    error_distances = []
    last_distance = 0

    for i, row in enumerate(window[0]):
        X = row[0]
        y = row[1]
        p = row[2]
        e = row[3]
        for fi, f in enumerate(X):
            features[fi].append(f)
        labels.append(y)
        predictions.append(p)
        errors.append(e)
        if not e:
            distance = i - last_distance
            error_distances.append(distance)
            last_distance = i
    if len(error_distances) == 0:
        error_distances = [0]

    return (features, labels, predictions, errors, error_distances)


def get_concept_stats(timeseries, model):
    concept_stats = {}

    features = timeseries[0]
    labels = timeseries[1]
    predictions = timeseries[2]
    errors = timeseries[3]
    error_distances = timeseries[4]

    length = len(features[0])
    X = []
    for i in range(length):
        X.append([])
        for f in features:
            X[-1].append(f[i])
        X[-1] = np.array(X[-1])
    X = np.array(X)

    shaps = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", check_additivity=False).shap_values(X, check_additivity=False)
    # If there is only 1 label, shaps just returns the matrix, otherwise it returns
    # a list of matricies. This converts the single case into a list.
    if not isinstance(shaps, list):
        shaps = [shaps]
    mean_shaps = np.sum(np.abs(shaps[0]), axis=0)
    SHAP_vals = [abs(x) for x in mean_shaps]
    for f1, f in enumerate(features):
        stats = get_timeseries_stats(f, SHAP_vals[f1])
        concept_stats[f"f{f1}"] = stats

    stats = get_timeseries_stats(list(map(float, labels)))
    concept_stats["labels"] = stats
    stats = get_timeseries_stats(list(map(float, predictions)))
    concept_stats["predictions"] = stats
    stats = get_timeseries_stats(list(map(float, errors)))
    concept_stats["errors"] = stats
    stats = get_timeseries_stats(list(map(float, error_distances)))
    concept_stats["error_distances"] = stats
    return concept_stats


def turningpoints(lst):
    dx = np.diff(lst)
    return np.sum(dx[1:] * dx[:-1] < 0)


def get_timeseries_stats(timeseries, FI=None,):
    stats = {}
    if len(timeseries) < 3:
        stats = {}
        stats["mean"] = statistics.mean(timeseries)
        stats["stdev"] = 0
        stats["skew"] = 0
        stats['kurtosis'] = 0
        if len(timeseries) > 1:
            stats["stdev"] = statistics.stdev(timeseries)
            stats["skew"] = scipy.stats.skew(timeseries)
            stats['kurtosis'] = scipy.stats.kurtosis(timeseries)
        stats["turning_point_rate"] = 0
        stats["acf_1"] = 0
        stats["acf_2"] = 0
        stats["pacf_1"] = 0
        stats["pacf_2"] = 0
        stats["MI"] = 0
        stats["FI"] = 0
        stats["IMF_0"] = 0
        stats["IMF_1"] = 0
        stats["IMF_2"] = 0
        return stats

    # emd = EMD(DTYPE=np.float16, max_imf=2)
    emd = EMD(max_imf=2, spline_kind='slinear')
    # IMFs = emd(np.array(timeseries, dtype=np.float16), max_imf=2)
    IMFs = emd(np.array(timeseries), max_imf=2)
    for i, imf in enumerate(IMFs):
        stats[f"IMF_{i}"] = perm_entropy(imf)
    for i in range(3):
        if f"IMF_{i}" not in stats:
            stats[f"IMF_{i}"] = 0
    stats["mean"] = statistics.mean(timeseries)
    stats["stdev"] = statistics.stdev(timeseries)
    stats["skew"] = scipy.stats.skew(timeseries)
    stats['kurtosis'] = scipy.stats.kurtosis(timeseries)
    tp = int(turningpoints(timeseries))
    tp_rate = tp / len(timeseries)
    stats['turning_point_rate'] = tp_rate

    try:
        acf_vals = acf(timeseries, nlags=3, fft=True)
    except Exception as e:
        print(e)
        exit()
        acf_vals = [-1 for x in range(6)]
    for i, v in enumerate(acf_vals):
        if i == 0:
            continue
        if i > 2:
            break
        stats[f"acf_{i}"] = v if not np.isnan(v) else -1

    try:
        acf_vals = pacf(timeseries, nlags=3)
    except Exception as e:
        print(e)
        acf_vals = [-1 for x in range(6)]
    for i, v in enumerate(acf_vals):
        if i == 0:
            continue
        if i > 2:
            break
        stats[f"pacf_{i}"] = v if not np.isnan(v) else -1

    if len(timeseries) > 4:
        current = np.array(timeseries)
        previous = np.roll(current, -1)
        current = current[:-1]
        previous = previous[:-1]
        X = np.array(current).reshape(-1, 1)
        MI = mutual_info_regression(X, previous)[0]
    else:
        MI = 0
    stats["MI"] = MI

    stats["FI"] = FI if FI is not None else 0

    return stats
# def construct_fingerprints_for_concepts()


def construct_sliding_window_stats(stream, output_path, options):
    window_size = options['window_size']
    observation_gap = options['observation_gap']
    stats_path = output_path / f"{window_size}_{observation_gap}_{options['train_len']}_stats"
    accuracy_path = output_path / f"{window_size}_{observation_gap}_accuracy"
    if stats_path.exists() and accuracy_path.exists():
        with stats_path.open('rb') as f:
            stats = pickle.load(f)
        with accuracy_path.open('rb') as f:
            acc_obj = json.load(f)
            right = acc_obj['right']
            wrong = acc_obj['wrong']
    else:
        normalizer = options['normalizer']
        print(normalizer.classes)
        print(options['classes'])
        classes_as_int = isinstance(normalizer.classes[0], (int, np.integer))
        print(classes_as_int)
        print(type(normalizer.classes[0]))
        all_classes = list(set(normalizer.classes))
        if options['classes'] is not None:
            all_classes = list(set(normalizer.classes + [int(x) if classes_as_int else x for x in options['classes']]))
        print(all_classes)
        # exit()
        detector = ADWIN()
        classifier = HoeffdingTree()
        stats = []
        window = deque(maxlen=window_size)
        stream.prepare_for_use()
        right = 0
        wrong = 0
        for i in tqdm.tqdm(range(options['length'] - 1)):
            X, y = stream.next_sample()
            p = classifier.predict(X)
            e = y[0] == p[0]
            right += y[0] == p[0]
            wrong += y[0] != p[0]
            classifier.partial_fit(X, y, all_classes)



            window.append((X[0], y[0], p[0], e))

            if i > window_size:
                if i % observation_gap == 0:
                    current_timeseries = window_to_timeseries([window, "n"])
                    concept_stats = get_concept_stats(current_timeseries,
                                                      classifier)
                    normalizer.add_stats(concept_stats)
                    stats.append((concept_stats, i))
            detector.add_element(e)
            if detector.detected_change():
                detector = ADWIN()
                classifier = HoeffdingTree()

        print(f"Accuracy: {right / (right + wrong)}")

        with stats_path.open('wb') as f:
            pickle.dump(stats, f)
        with accuracy_path.open('w') as f:
            json.dump({"accuracy": right / (right + wrong),
                       'right': int(right),
                       'wrong': int(wrong)}, f)
    return stats, right, wrong


def get_flat_vector(stats, normalizer, options, skip_sources=[]):
    vec = []
    skip_sources = set(skip_sources)
    for source in normalizer.sources:
        if source in skip_sources:
            continue
        for feature in normalizer.features:
            if options['meta-features'][feature] != 1:
                continue
            value = stats[source][feature]
            normed_value = normalizer.get_normed_value(source, feature, value)
            vec.append(normed_value)
    if len(vec) == 0:
        raise ValueError("No vector created, check at least one meta-feature enabled in options.")
    return np.array(vec)


def get_source_vector(stats, normalizer, options, skip_sources=[]):
    vec = []
    skip_sources = set(skip_sources)
    for source in normalizer.sources:
        if source in skip_sources:
            continue
        source_vec = []
        for feature in normalizer.features:
            if options['meta-features'][feature] != 1:
                continue
            value = stats[source][feature]
            normed_value = normalizer.get_normed_value(source, feature, value)
            source_vec.append(normed_value)
        vec.append(np.array(source_vec))
    return np.array(vec)


def get_feature_vector(stats, normalizer, options, skip_sources=[]):
    return np.transpose(get_source_vector(stats, normalizer, options, skip_sources))

def get_distance(A, B, distance_calc, distance_measure, vector_type, normalizer, options):
    supervised_sources = ['labels', 'errors', 'error_distances']

    skip_sources = []
    if vector_type == "U":
        skip_sources = supervised_sources
    
    if distance_measure == 'cosine':
        dist_func = get_cosine_distance
    elif distance_measure == 'pearson':
        dist_func = get_pearson_distance
    else:
        raise ValueError

    if distance_calc == 'flat':
        return get_flat_distance(A, B, normalizer, options, dist_func, vector_type)
    elif distance_calc == 'feature':
        return get_feature_distance(A, B, normalizer, options, dist_func, vector_type)
    elif distance_calc == 'source':
        return get_source_distance(A, B, normalizer, options, dist_func, vector_type)
    else:
        raise ValueError
    

def get_cosine_distance(A, B):
    try:
        c = cosine(A, B) 
    except:
        c = np.nan
    if np.isnan(c):
        c = 0 if ((not np.any(A)) and (not np.any(B))) else 1
        # print('error')
    return c


def get_pearson_distance(A, B):
    try:
        p = pearsonr(A, B)
    except:
        p = [np.nan]
    if np.isnan(p[0]):
        A_is_constant = np.all(A == A[0])
        B_is_constant = np.all(B == B[0])
        p = [0] if (A_is_constant and B_is_constant) else [1]
        # print('error')
    return 1 - p[0]

def get_flat_distance(A, B, normalizer, options, dist_func, vector_type):
    # A_flat = get_flat_vector(A, normalizer, options, skip_sources)
    # B_flat = get_flat_vector(B, normalizer, options, skip_sources)
    # print(A)
    A_flat = A[vector_type]['flat']
    B_flat = B[vector_type]['flat']
    d = dist_func(A_flat, B_flat)
    # if np.isnan(d):
    #     print('error')
    return d

def get_feature_distance(A, B, normalizer, options, dist_func, vector_type):
    A_by_feature = A[vector_type]['feature']
    B_by_feature = B[vector_type]['feature']
    # A_by_feature = get_feature_vector(A, normalizer, options, skip_sources)
    # B_by_feature = get_feature_vector(B, normalizer, options, skip_sources)

    num_features, num_sources = A_by_feature.shape
    avg_distance = 0
    for f in range(num_features):
        A_feature = A_by_feature[f]
        B_feature = B_by_feature[f]

        distance = dist_func(A_feature, B_feature)
        avg_distance += distance
    d = avg_distance / num_features
    # if np.isnan(d):
    #     print('error')
    return d


def get_source_distance(A, B, normalizer, options, dist_func, vector_type):
    A_by_source = A[vector_type]['source']
    B_by_source = B[vector_type]['source']
    # A_by_source = get_source_vector(A, normalizer, options, skip_sources)
    # B_by_source = get_source_vector(B, normalizer, options, skip_sources)

    num_sources, num_features = A_by_source.shape
    avg_distance = 0
    for f in range(num_sources):
        A_source = A_by_source[f]
        B_source = B_by_source[f]

        distance = dist_func(A_source, B_source)
        avg_distance += distance
    d = avg_distance / num_sources
    # if np.isnan(d):
    #     print('error')
    return d


def get_vector_types(stats, normalizer, options):
    supervised_sources = ['labels', 'errors', 'error_distances']
    vector_types = {}
    for vector_type in ['S', 'U']:
        skip_sources = []
        if vector_type == "U":
            skip_sources = supervised_sources

        vectors = {
            'flat': get_flat_vector(stats, normalizer, options, skip_sources),
            'feature': get_feature_vector(stats, normalizer, options, skip_sources),
            'source': get_source_vector(stats, normalizer, options, skip_sources),
        }
        vector_types[vector_type] = vectors
    return vector_types

def get_weighted_cosine_distance(stats_observation, comparison_stats, normalizer, stat_weights, options, skip_sources = []):
    observation_vec = []
    comparison_vec = []
    weights = []
    skip_sources = set(skip_sources)
    for source in normalizer.sources:
        if source in skip_sources:
            continue
        for feature in normalizer.features:
            if options['meta-features'][feature] != 1:
                continue
            observation_value = stats_observation[source][feature]
            normed_observation_value = normalizer.get_normed_value(source, feature, observation_value)
            observation_vec.append(normed_observation_value)

            comparison_value = comparison_stats[source][feature]
            normed_comparison_value = normalizer.get_normed_value(source, feature, comparison_value)
            comparison_vec.append(normed_comparison_value)

            weight = stat_weights[source][feature]
            weights.append(weight)
    if len(observation_vec) == 0:
        raise ValueError("No vector created, check at least one meta-feature enabled in options.")

    weights = np.array(weights)
    weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weight))
    try:
        c = cosine(observation_vec, comparison_vec, w=weights) 
    except:
        c = np.nan
    if np.isnan(c):
        c = 0 if ((not np.any(A)) and (not np.any(B))) else 1
        print('error')
    return c






def get_stream_similarity(stream_stats, concept_stats, normalizer, options, stat_weights = None):
    similarity = {}
    comparison_vec_cache = {}
    for stats_observation, timestep in tqdm.tqdm(stream_stats):

        for comparison_concept in concept_stats:
            if comparison_concept not in similarity:
                similarity[comparison_concept] = []
            comparison_stats = concept_stats[comparison_concept]

            observation_vectors = get_vector_types(stats_observation, normalizer, options)
            if comparison_concept in comparison_vec_cache:
                comparison_vectors = comparison_vec_cache[comparison_concept]
            else:
                comparison_vectors = get_vector_types(comparison_stats, normalizer, options)
                comparison_vec_cache[comparison_concept] = comparison_vectors
            sim = {'s_cosign': get_distance(observation_vectors, comparison_vectors, 'flat', 'cosine', 'S', normalizer, options),
                   'u_cosine': get_distance(observation_vectors, comparison_vectors, 'flat', 'cosine', 'U', normalizer, options),
                   's_pearson': get_distance(observation_vectors, comparison_vectors, 'flat', 'pearson', 'S', normalizer, options),
                   'u_pearson': get_distance(observation_vectors, comparison_vectors, 'flat', 'pearson', 'U', normalizer, options),
                   'cs_by_feature': get_distance(observation_vectors, comparison_vectors, 'feature', 'cosine', 'S', normalizer, options),
                   'cu_by_feature': get_distance(observation_vectors, comparison_vectors, 'feature', 'cosine', 'U', normalizer, options),
                   'cs_by_source': get_distance(observation_vectors, comparison_vectors, 'source', 'cosine', 'S', normalizer, options),
                   'cu_by_source': get_distance(observation_vectors, comparison_vectors, 'source', 'cosine', 'U', normalizer, options),
                   'ps_by_feature': get_distance(observation_vectors, comparison_vectors, 'feature', 'pearson', 'S', normalizer, options),
                   'pu_by_feature': get_distance(observation_vectors, comparison_vectors, 'feature', 'pearson', 'U', normalizer, options),
                   'ps_by_source': get_distance(observation_vectors, comparison_vectors, 'source', 'pearson', 'S', normalizer, options),
                   'pu_by_source': get_distance(observation_vectors, comparison_vectors, 'source', 'pearson', 'U', normalizer, options)}
            if stat_weights is not None:
                sim['weighted_s_cosine'] = get_weighted_cosine_distance(stats_observation, comparison_stats, normalizer, stat_weights, options )
            # print(sim)
            similarity[comparison_concept].append((sim, timestep))
    return similarity


def get_metafeature_name(options):
    metafeatures = options['meta-features']

    kv_list = [(k, v) for k, v in metafeatures.items()]

    name = '-'.join([f"{k[:3]}${v}" for k, v in kv_list])
    return f"{name}-sf${'$'.join(options['sampler_features'][:3])}"


def process_item(options):
    storage_path = pathlib.Path.cwd() / "storage" / options['data_name']
    output_path = pathlib.Path.cwd() / "output" / options['data_name'] / get_metafeature_name(options)
    print(f"Running {output_path}")
    results_filename = output_path / f"{options['data_name'] }_results.txt"
    if options['overwrite'] and output_path.exists():
        shutil.rmtree(output_path)
    if results_filename.exists():
        if not options['replace_output']:
            print(f"{output_path} skipped")
            return
        replacement_try = 1
        replacement_filename = output_path.parent / f"{output_path.stem}_O{replacement_try}"
        while replacement_filename.exists():
            replacement_try += 1
            replacement_filename = replacement_filename.parent / f"{replacement_filename.stem}_O{replacement_try}"

        shutil.move(output_path, replacement_filename)
    if not storage_path.exists():
        storage_path.mkdir()

    if 'normalizer' not in options or options['normalizer'] is None:
        normalizer_path = storage_path / f"{options['train_len']}_normalizer"
        if normalizer_path.exists():
            with normalizer_path.open('rb') as f:
                normalizer = pickle.load(f)
        else:
            normalizer = Normalizer()
            
    else:
        normalizer = options['normalizer']

    if 'concept_stats' not in options or options['concept_stats'] is None:
        if options['data_type'] == "synthetic":
            concepts = load_synthetic_concepts(options['data_name'],
                                            options['data_source'], 0,
                                            sampler_features = options['sampler_features'])
        else:
            concepts = load_real_concepts(options['data_name'],
                                        options['data_source'], 0,
                                      nrows = options['nrows'])
        for concept_gen, _ in concepts:
            concept_gen.prepare_for_use()

        windows, skipped = train_concept_windows(concepts, normalizer, storage_path, options)
        concept_stats = average_concept_stats(windows)
    else:
        concept_stats = options['concept_stats']

    if 'stats' not in options or options['stats'] is None:    
        if options['data_type'] == "synthetic":
            concepts = load_synthetic_concepts(options['data_name'],
                                            options['data_source'], 0,
                                            sampler_features = options['sampler_features'])
        else:
            concepts = load_real_concepts(options['data_name'],
                                        options['data_source'], 0,
                                      nrows = options['nrows'])

        stream_concepts, length = get_inorder_concept_ranges(concepts)
        stream = AbruptDriftStream(stream_concepts, length)

        stats, right, wrong = construct_sliding_window_stats(stream,
                                                            storage_path,
                                                            options)
    else:
        stats = options['stats']
        stream_concepts = options['stream_concepts']
        length = options['length']

    # with normalizer_path.open('rb') as f:
    #     stored_normalizer = pickle.load(f)

    # normalizer.merge(stored_normalizer)
    # with normalizer_path.open('wb') as f:
    #     pickle.dump(normalizer, f)

    # profiler = Profiler()
    # profiler.start()
    sims = get_stream_similarity(stats, concept_stats, normalizer, options)
    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))
 
    if not output_path.exists():
        print(output_path)
        print(output_path.exists())
        print(os.path.exists(output_path))
        time.sleep(0.5)
        output_path.mkdir(parents=True, exist_ok = True)
    print(output_path.exists())
    with open(output_path / f"{options['data_name']}_stats.pickle", 'wb+') as f:
        pickle.dump(stats, f)
    with open(output_path / f"{options['data_name']}_sims.pickle", 'wb') as f:
        pickle.dump(sims, f)
    with open(output_path / f"{options['data_name']}_concepts.pickle", 'wb') as f:
        pickle.dump(stream_concepts, f)
    with open(output_path / f"{options['data_name']}_length.pickle", 'wb') as f:
        pickle.dump(length, f)
    with open(output_path / f"{options['data_name']}_options.txt", 'w') as f:
        json.dump(options, f, default=lambda o: '<not serializable>')

    losses, all_distance_avg_loss = run_eval(stats, sims, stream_concepts, length, options['data_name'], window_size = options['window_size'], path=output_path, show=False)
    results = {
        'losses_by_distance_measure': losses,
        'all_distance_avg_loss': all_distance_avg_loss
    }
    
    with open(str(results_filename), 'w') as f:
        json.dump(results, f)
    print(all_distance_avg_loss)

def remake_normalizer(windows):
    print("remaking normalizer")
    normalizer = Normalizer()
    for w in windows:
        print(w)
        observations = w[0]
        for o in observations:
            y = o[1]
            print(y)
            normalizer.add_class(y)
        stats = w[3]
        print(stats)
        normalizer.add_stats(stats)
    return normalizer


def powerset(iterable, max_len = None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    max_len = max_len if max_len is not None else len(s) + 1
    return chain.from_iterable(combinations(s, r) for r in range(1, min(max_len+1, len(s)+1)))

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--name', default='Arabic', type=str)
    my_parser.add_argument('--syn', action='store_true')
    my_parser.add_argument('--single', action='store_true')
    my_parser.add_argument('--overwrite', action='store_true')
    my_parser.add_argument('--replaceoutput', action='store_true')
    my_parser.add_argument('--cpu', default=2, type=int)
    my_parser.add_argument('--nrows', default=-1, type=int, help="Number of rows of real\
                                data to read. -1 for all (default)")
    my_parser.add_argument('--classes', nargs='*', help='We try to detect classes automatically\
                                when the normalizer is set up, but sometimes this does not find\
                                rare classes. In this case, manually pass all clases in the dataset.')

    args = my_parser.parse_args()
    options = {
        'data_name': args.name,
        'data_type': 'real',
        'data_source': 'Real',
        'classes': args.classes,
        'nrows': args.nrows,
        # 'window_size': 60,
        # 'window_size': 600,
        'window_size': -1,
        'overwrite': args.overwrite,
        'replace_output': args.replaceoutput,
        'observation_gap': -1,
        'train_len': 2500,
        'sampler_features': ['distribution', 'autocorrelation', 'frequency'],
        # 'sampler_features': ['distribution', 'frequency'],
        # 'sampler_features': ['distribution', 'autocorrelation'],
        # 'sampler_features': ['frequency', 'autocorrelation'],
        # 'sampler_features': ['distribution'],
        'meta-features': {
            "mean": 0,
            "stdev": 0,
            "skew": 0,
            "kurtosis": 0,
            "turning_point_rate": 0,
            "acf_1": 0,
            "acf_2": 0,
            "pacf_1": 0,
            "pacf_2": 0,
            "MI": 0,
            "FI": 0,
            "IMF_0": 0,
            "IMF_1": 0,
            "IMF_2": 0,
        }
    }
    if args.syn:
        options['data_type'] = 'synthetic'
        options['data_source'] = args.name

    storage_path = pathlib.Path.cwd() / "storage" / options['data_name']
    if options['overwrite'] and storage_path.exists():
        shutil.rmtree(storage_path)
    if not storage_path.exists():
        storage_path.mkdir()
        

    if options['data_type'] == "synthetic":
        concepts = load_synthetic_concepts(options['data_name'],
                                           options['data_source'], 0,
                                           overwrite = options['overwrite'],
                                            sampler_features = options['sampler_features'])
    else:
        concepts = load_real_concepts(options['data_name'],
                                      options['data_source'], 0,
                                      overwrite = options['overwrite'],
                                      nrows = options['nrows'])
    for concept_gen, _ in concepts:
        concept_gen.prepare_for_use()
    stream_concepts, length = get_inorder_concept_ranges(concepts)
    options['length'] = length
    if options['window_size'] == -1:
        options['window_size'] = round(length / 15)
    if options['observation_gap'] == -1:
        options['observation_gap'] = math.ceil(length / 1500)
    normalizer_path = storage_path / f"{options['train_len']}_normalizer"
    if normalizer_path.exists():
        with normalizer_path.open('rb') as f:
            normalizer = pickle.load(f)
    else:
        normalizer = Normalizer()
    windows, skipped = train_concept_windows(concepts, normalizer, storage_path, options)
    if skipped:
        normalizer = remake_normalizer(windows)
        with normalizer_path.open('wb') as f:
            pickle.dump(normalizer, f)
    concept_stats = average_concept_stats(windows)

    if options['data_type'] == "synthetic":
        concepts = load_synthetic_concepts(options['data_name'],
                                           options['data_source'], 0,
                                            sampler_features = options['sampler_features'])
    else:
        concepts = load_real_concepts(options['data_name'],
                                      options['data_source'], 0,
                                      nrows = options['nrows'])

    stream_concepts, length = get_inorder_concept_ranges(concepts)
    stream = AbruptDriftStream(stream_concepts, length)
    options['normalizer'] = normalizer
    options['concept_stats'] = concept_stats

    stats, right, wrong = construct_sliding_window_stats(stream,
                                                         storage_path,
                                                         options)

    options['stats'] = stats
    options['stream_concepts'] = stream_concepts
    options['length'] = length

    if not args.single:
        MF_options = ['mean', 'stdev', 'skew', 'kurtosis', 'turning_point_rate', 'acf', 'pacf', 'MI', 'FI', 'IMF']
        MF_combinations = list(powerset(MF_options))
        print(MF_combinations)
        print(len(MF_combinations))
        option_combos = []
        for MFs in MF_combinations:
            for k in options['meta-features']:
                k_no_suffix = k[:-2]
                k_in_MF = any([x == k or x == k_no_suffix for x in MFs])
                k_considered = any([x == k or x == k_no_suffix for x in MF_options])
                options['meta-features'][k] = 1 if (k_in_MF or not k_considered) else 0
            combo = {**options, 'meta-features': {**options['meta-features']}}
            option_combos.append(combo)

        pool = mp.Pool(processes=args.cpu)
        results = pool.map(process_item, option_combos)
    else:
        options['meta-features'] = {
            "mean": 1,
            "stdev": 1,
            "skew": 1,
            "kurtosis": 1,
            "turning_point_rate": 1,
            "acf_1": 1,
            "acf_2": 1,
            "pacf_1": 1,
            "pacf_2": 1,
            "MI": 1,
            "FI": 1,
            "IMF_0": 1,
            "IMF_1": 1,
            "IMF_2": 1,
        }

        process_item(options)

        

    # process_item(options)
    
