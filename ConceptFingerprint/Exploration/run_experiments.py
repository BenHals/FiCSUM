import pickle
import json
import pathlib
import cProfile
from itertools import chain, combinations
import multiprocessing as mp



import tqdm
import numpy as np

from ConceptFingerprint.Data.load_data import load_synthetic_concepts, load_real_concepts, get_inorder_concept_ranges, AbruptDriftStream
from ConceptFingerprint.Exploration.sliding_window_MI import learn_fingerprints_from_concepts, get_sliding_window_similarities, Normalizer, get_metafeature_name, train_concept_windows
from ConceptFingerprint.Exploration.explore_SW import get_data, run_eval

def run(name, data_type, data_source, output_path, options):

    # name = "cmc"
    # data_type = "real"
    # data_source = "Real"

    if data_type == "synthetic":
        concepts = load_synthetic_concepts(options['data_name'], options['data_source'], 0)
    else:
        concepts = load_real_concepts(options['data_name'], options['data_source'], 0)
    train_len = 10000
    storage_path = pathlib.Path.cwd() / "storage" / options['data_name']
    normalizer_path = storage_path / f"{train_len}_normalizer"
    fingerprint_path = storage_path / f"{train_len}_fingerprints"
    fingerprint_stats_path = storage_path / f"{train_len}_fingerprints_stats"

    with normalizer_path.open('rb') as f:
        normalizer = pickle.load(f)
    with fingerprint_path.open('rb') as f:
        fingerprints = pickle.load(f)
    with fingerprint_stats_path.open('rb') as f:
        fingerprint_stats = pickle.load(f)
    # normalizer = Normalizer()
    # fingerprints, fingerprint_stats, normalizer = learn_fingerprints_from_concepts(concepts, normalizer, options)

    if data_type == "synthetic":
        concepts = load_synthetic_concepts(name, data_source, 0)
    else:
        concepts = load_real_concepts(name, data_source, 0)

    stream_concepts, length = get_inorder_concept_ranges(concepts)
    stream = AbruptDriftStream(stream_concepts, length)

    stats, sims, right, wrong = get_sliding_window_similarities(stream, length, fingerprints, normalizer, options=options)

    if not output_path.exists():
        output_path.mkdir()

    with open(str(output_path / f"{name}_stats.pickle"), 'wb') as f:
        pickle.dump(stats, f)
    with open(str(output_path / f"{name}_sims.pickle"), 'wb') as f:
        pickle.dump(sims, f)
    with open(str(output_path / f"{name}_concepts.pickle"), 'wb') as f:
        pickle.dump(stream_concepts, f)
    with open(str(output_path / f"{name}_length.pickle"), 'wb') as f:
        pickle.dump(length, f)
    with open(str(output_path / f"{name}_options.txt"), 'w') as f:
        json.dump(options, f)

    return run_eval(stats, sims, stream_concepts, length, name, path=output_path, show=False)

def powerset(iterable, max_len = None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    max_len = max_len if max_len is not None else len(s) + 1
    return chain.from_iterable(combinations(s, r) for r in range(1, min(max_len+1, len(s)+1)))

def process_options(options):
    print(options['meta-features'])
    output_path = pathlib.Path.cwd() / "output" / options['data_name'] / get_metafeature_name(options)
    results_filename = output_path / f"{options['data_name'] }_results.txt"
    if results_filename.exists():
        return
    losses, all_distance_avg_loss = run(options['data_name'], options['data_type'], options['data_source'], output_path, options)
    results = {
        'losses_by_distance_measure': losses,
        'all_distance_avg_loss': all_distance_avg_loss
    }
    with open(str(results_filename), 'w') as f:
        json.dump(results, f)
    print(all_distance_avg_loss)


def main():
    name = "qg"
    data_type = "real"
    data_source = "Real"
    options = {
        'data_name': 'qg',
        'data_type': 'real',
        'data_source': 'Real',
        'window_size': 60,
        'observation_gap': 15,
        'meta-features': {
            "mean": 1,
            "stdev": 1,
            "skew": 0,
            "kurtosis": 0,
            "turning_point_rate": 0,
            "acf": 0,
            "pacf": 0,
            "MI": 0,
            "FI": 0,
            "IMF": 0,
        }
    }

    MF_options = ['skew', 'kurtosis', 'turning_point_rate', 'acf', 'pacf', 'MI', 'FI', 'IMF']
    MF_combinations = list(powerset(MF_options))
    print(MF_combinations)
    print(len(MF_combinations))
    train_len = 10000
    storage_path = pathlib.Path.cwd() / "storage" / options['data_name']
    normalizer_path = storage_path / f"{train_len}_normalizer"
    fingerprint_path = storage_path / f"{train_len}_fingerprints"
    fingerprint_stats_path = storage_path / f"{train_len}_fingerprints_stats"
    if not normalizer_path.exists():
        if data_type == "synthetic":
            concepts = load_synthetic_concepts(options['data_name'], options['data_source'], 0)
        else:
            concepts = load_real_concepts(options['data_name'], options['data_source'], 0)
        for concept_gen, _ in concepts:
            concept_gen.prepare_for_use()
        normalizer = Normalizer()
        train_concept_windows(concepts, normalizer, options = options)
    
    if not fingerprint_path.exists():
        if data_type == "synthetic":
            concepts = load_synthetic_concepts(options['data_name'], options['data_source'], 0)
        else:
            concepts = load_real_concepts(options['data_name'], options['data_source'], 0)
        with normalizer_path.open('rb') as f:
            normalizer = pickle.load(f)
        fingerprints, fingerprint_stats, normalizer = learn_fingerprints_from_concepts(concepts, normalizer, options)
        with normalizer_path.open('wb') as f:
            pickle.dump(normalizer, f)
        with fingerprint_path.open('wb') as f:
            pickle.dump(fingerprints, f)
        with fingerprint_stats_path.open('wb') as f:
            pickle.dump(fingerprint_stats, f)

    option_combos = []
    option_combos = [{
        'data_name': 'qg',
        'data_type': 'real',
        'data_source': 'Real',
        'window_size': 60,
        'observation_gap': 15,
        'meta-features': {
            "mean": 1,
            "stdev": 1,
            "skew": 1,
            "kurtosis": 1,
            "turning_point_rate": 1,
            "acf": 1,
            "pacf": 1,
            "MI": 1,
            "FI": 1,
            "IMF": 0,
        }
    }]
    # for MFs in MF_combinations:
    #     for k in options['meta-features']:
    #         options['meta-features'][k] = 1 if (k in MFs or k not in MF_options) else 0
    #     option_combos.append({**options, 'meta-features': {**options['meta-features']}})
    print(option_combos)
    pool = mp.Pool(processes=3)
    results = pool.map(process_options, option_combos)
    # process_options(options)

if __name__ == "__main__":
    main()