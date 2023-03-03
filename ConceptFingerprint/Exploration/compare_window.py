# from conceptfingerprint.Data.load_data import *
import statistics
import scipy.stats

from ConceptFingerprint.Data.load_data import load_synthetic_concepts, load_real_concepts

from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats.stats import pearsonr

class Normalizer:
    def __init__(self):
        self.S = []
        self.U = []
    
    def add_observation(self, S, U):
        for i, s_feature in enumerate(S):
            if i >= len(self.S):
                self.S.append([None, None])
            min_f = self.S[i][0]
            max_f = self.S[i][1]
            if min_f is None or min_f > s_feature:
                self.S[i][0] = s_feature
            if max_f is None or max_f < s_feature:
                self.S[i][1] = s_feature
        for i, s_feature in enumerate(U):
            if i >= len(self.U):
                self.U.append([None, None])
            min_f = self.U[i][0]
            max_f = self.U[i][1]
            if min_f is None or min_f > s_feature:
                self.U[i][0] = s_feature
            if max_f is None or max_f < s_feature:
                self.U[i][1] = s_feature
    
    def get_normed_vec(self, f, f_type = None):
        if f_type is None:
            f_type = "U" if len(f) < len(self.S) else "S"
        
        normalizer = self.S if f_type == "S" else self.U

        normed_f = []
        for i,feature in enumerate(f):
            f_range = normalizer[i]
            width = (f_range[1] - f_range[0]) if (f_range[1] - f_range[0]) != 0 else max(f_range[1], 1) / 10
            normed = (f_range[1] - feature) / width
            normed_f.append(normed)
        
        return normed_f


def trained_concept_window_meta_info(concepts, extractors, window_size = 200):
    windows = []
    for concept_gen, concept_name in concepts:
        windows.append(([], concept_name))

        model = HoeffdingTree()
        for i in range(1000):
            print(concept_gen.n_remaining_samples())
            X,y = concept_gen.next_sample()
            model.partial_fit(X, y)


        for i in range(window_size):
            print(concept_gen.n_remaining_samples())
            if concept_gen.n_remaining_samples() == 0:
                concept_gen.restart()
            X,y = concept_gen.next_sample()
            p = model.predict(X)
            # print(X)
            # print(y)
            # print(p)
            e = p[0] == y[0]
            windows[-1][0].append((X[0],y[0], p[0], e))
    return windows
def random_concept_window_meta_info(concepts, extractors, window_size = 200, num_concepts = 20):
    windows = []
    print(concepts)
    # for concept_gen, concept_name in concepts:
    #     concept_gen.prepare_for_use()
    for ci in np.random.choice(len(concepts), num_concepts):
        concept_gen, concept_name = concepts[ci]
        windows.append(([], concept_name))

        model = HoeffdingTree()
        for i in range(1000):
            print(concept_gen.n_remaining_samples())
            if concept_gen.n_remaining_samples() == 0:
                concept_gen.restart()
            X,y = concept_gen.next_sample()
            model.partial_fit(X, y)


        for i in range(window_size):
            print(concept_gen.n_remaining_samples())
            if concept_gen.n_remaining_samples() == 0:
                concept_gen.restart()
            X,y = concept_gen.next_sample()
            p = model.predict(X)
            e = p[0] == y[0]
            windows[-1][0].append((X[0],y[0], p[0], e))
    return windows

def window_to_timeseries(window):
    # print(window)
    features = []
    for f in window[0][0][0]:
        features.append([])
    labels = []
    predictions = []
    errors = []
    error_distances = []
    last_distance = 0

    for i,row in enumerate(window[0]):
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

def get_timeseries_stats(timeseries):
    # print(timeseries)
    stats = {}
    if len(timeseries) < 3:
        return {
            "mean": 0,
            "stdev": 0,
            "skew": 0,
            "kurtosis": 0,
            "acf_1": 0,
            "acf_2": 0,
            "pacf_1": 0,
            "pacf_2": 0,
        }
    stats["mean"] = statistics.mean(timeseries)
    stats["stdev"] = statistics.stdev(timeseries)
    stats["skew"] = scipy.stats.skew(timeseries)
    stats['kurtosis'] = scipy.stats.kurtosis(timeseries)
    try:
        acf_vals = acf(timeseries, nlags = 5)
    except:
        acf_vals = [-1 for x in range(6)]
    for i,v in enumerate(acf_vals):
        if i == 0: continue
        if i > 2: break
        stats[f"acf_{i}"] = v if not np.isnan(v) else -1
    try:
        acf_vals = pacf(timeseries, nlags = 5)
    except:
        acf_vals = [-1 for x in range(6)]
    for i,v in enumerate(acf_vals):
        if i == 0: continue
        if i > 2: break
        stats[f"pacf_{i}"] = v if not np.isnan(v) else -1

    return stats

def get_concept_stats(timeseries):
    concept_stats = {}
    s_fingerprint_vector = []
    u_fingerprint_vector = []

    features = timeseries[0]
    labels = timeseries[1]
    predictions = timeseries[2]
    errors = timeseries[3]
    error_distances = timeseries[4]
    print("features")
    for f1, f in enumerate(features):
        stats = get_timeseries_stats(f)
        concept_stats[f"f{f1}"] = stats
        s_fingerprint_vector = [*s_fingerprint_vector, *list(stats.values())]
        u_fingerprint_vector = [*u_fingerprint_vector, *list(stats.values())]
    print("labels")
    stats = get_timeseries_stats(list(map(float, labels)))
    concept_stats["labels"] = stats
    s_fingerprint_vector = [*s_fingerprint_vector, *list(stats.values())]
    print("predictions")
    stats = get_timeseries_stats(list(map(float, predictions)))
    concept_stats["predictions"] = stats
    s_fingerprint_vector = [*s_fingerprint_vector, *list(stats.values())]
    u_fingerprint_vector = [*u_fingerprint_vector, *list(stats.values())]
    print("errors")
    stats = get_timeseries_stats(list(map(float, errors)))
    concept_stats["errors"] = stats
    s_fingerprint_vector = [*s_fingerprint_vector, *list(stats.values())]
    print("error_distances")
    stats = get_timeseries_stats(list(map(float, error_distances)))
    concept_stats["error_distances"] = stats
    s_fingerprint_vector = [*s_fingerprint_vector, *list(stats.values())]

    return concept_stats, s_fingerprint_vector, u_fingerprint_vector

def test_fingerprints(fingerprints, concepts, normalizer):
    windows = random_concept_window_meta_info(concepts, None)
    for w in windows:
        print(f"*******")
        print(f"Random Concept was {w[1]}")
        timeseries = window_to_timeseries(w)
        concept_stats, sfingerprint, ufingerprint = get_concept_stats(timeseries)
        normalizer.add_observation(sfingerprint, ufingerprint)
        min_similarity = None
        min_similarity_concept = None
        for sf, uf,n in fingerprints:
            # sim = cosine(normalizer.get_normed_vec(sf), normalizer.get_normed_vec(sfingerprint))
            sim = 1 - pearsonr(normalizer.get_normed_vec(sf, f_type = "S"), normalizer.get_normed_vec(sfingerprint, f_type= "S"))[0]
            print(f"S_Distance to {n} is {sim}")
            if min_similarity is None or sim < min_similarity:
                min_similarity = sim
                min_similarity_concept = n
        
        print(f"Most similar Concept to {w[1]} was {min_similarity_concept} using S")
        min_similarity = None
        min_similarity_concept = None
        for sf, uf, n in fingerprints:
            # sim = cosine(normalizer.get_normed_vec(uf), normalizer.get_normed_vec(ufingerprint))
            sim = 1 - pearsonr(normalizer.get_normed_vec(uf, f_type = "U"), normalizer.get_normed_vec(ufingerprint, f_type= "U"))[0]
            print(f"U_Distance to {n} is {sim}")
            if min_similarity is None or sim < min_similarity:
                min_similarity = sim
                min_similarity_concept = n
        
        print(f"Most similar Concept to {w[1]} was {min_similarity_concept} using U")
        print(f"*******")

concepts = load_synthetic_concepts("test", "STAGGER", 0)
# concepts = load_real_concepts("UCI-Wine", "Real", 0)
# concepts = load_synthetic_concepts("testARGWAL", "ARGWAL", 0)
# concepts = load_synthetic_concepts("testTREE", "RTREE", 0)
normalizer = Normalizer()
# concepts = load_synthetic_concepts("testSEA", "SEA", 0)
for concept_gen, _ in concepts:
    concept_gen.prepare_for_use()
fingerprints = []
stats = {}
SFs = {}
UFs = {}
for i in range(10):
    windows = trained_concept_window_meta_info(concepts, None)
    for w in windows:
        print(w[1])
        if w[1] not in stats:
            stats[w[1]] = {}
        if w[1] not in SFs:
            SFs[w[1]] = []
        if w[1] not in UFs:
            UFs[w[1]] = []
        timeseries = window_to_timeseries(w)
        concept_stats, sfingerprint, ufingerprint = get_concept_stats(timeseries)
        normalizer.add_observation(sfingerprint, ufingerprint)
        # print(concept_stats)
        # print(sfingerprint)
        # print(ufingerprint)
        SFs[w[1]].append(sfingerprint)
        UFs[w[1]].append(ufingerprint)

        for k in concept_stats.keys():
            if k not in stats[w[1]]:
                stats[w[1]][k] = {}
            for s in concept_stats[k].keys():
                if s not in stats[w[1]][k]:
                    stats[w[1]][k][s]= []
                stats[w[1]][k][s].append(concept_stats[k][s])

for concept in SFs.keys():
    fingerprints.append((np.mean(SFs[concept], axis = 0), np.mean(UFs[concept], axis = 0), concept))

    
test_fingerprints(fingerprints, concepts, normalizer)

compare = {}
for k in stats.keys():
    for f in stats[k]:
        if f not in compare:
            compare[f] = {}
        for s in stats[k][f]:
            if s not in compare[f]:
                compare[f][s] = []
            m = statistics.mean(stats[k][f][s])
            st = statistics.stdev(stats[k][f][s])
            compare[f][s].append((k, m, st))

print(compare)

    # print(labels)

    