import statistics
import pickle
import json
import pathlib
from collections import deque
from itertools import chain, combinations
import multiprocessing as mp
import argparse

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
        AbruptDriftStream,\
        load_real_datastream
from ConceptFingerprint.Exploration.MF_importance_experiment import \
    window_to_timeseries,\
    get_concept_stats,\
    Normalizer
from ConceptFingerprint.Exploration.sliding_window_MI import learn_fingerprints_from_concepts, get_sliding_window_similarities, get_metafeature_name
from ConceptFingerprint.Exploration.explore_SW import get_data, run_eval


name = "Elec"

stream = load_real_datastream(name, "real", 0)
stream.prepare_for_use()
length = stream.n_remaining_samples()
window_size = 60
observation_gap = 1
detector = ADWIN()
classifier = HoeffdingTree()
stats = []
window = deque(maxlen=window_size)
stream.prepare_for_use()
right = 0
wrong = 0
for i in tqdm.tqdm(range(length - 1)):
    X, y = stream.next_sample()
    p = classifier.predict(X)
    e = y[0] == p[0]
    right += y[0] == p[0]
    wrong += y[0] != p[0]
    classifier.partial_fit(X, y, normalizer.classes)



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
# print(stream.n_remaining_samples())