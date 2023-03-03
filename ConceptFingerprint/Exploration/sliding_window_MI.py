#%%
import statistics
from collections import deque
import pickle
import pathlib

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

from ConceptFingerprint.Data.load_data import load_synthetic_concepts, load_real_concepts, get_inorder_concept_ranges, AbruptDriftStream

class Normalizer:
    def __init__(self):
        self.S = []
        self.U = []
        self.classes = []
        self.features = []
        self.S_features = []

    def add_feature_observations(self, S, U):
        
        for a0 in range(S.shape[0]):
            if a0 >= len(self.S_features):
                self.S_features.append([])
            for a1 in range(S.shape[1]):
                if a1 >= len(self.S_features[a0]):
                    self.S_features[a0].append([None, None])
                
                val = S[a0][a1]
                min_f = self.S_features[a0][a1][0]
                max_f = self.S_features[a0][a1][1]
                if min_f is None or min_f > val:
                    self.S_features[a0][a1][0] = val
                if max_f is None or max_f < val:
                    self.S_features[a0][a1][1] = val

    # def add_feature_observations(self, S, U):
    #     num_features = S.shape[0]
    #     for f in range(num_features):
    #         print(S[f])
    #         print(U[f])
    #         vals = [*S[f], *U[f]]
    #         print(vals)
    #         max_val = max(vals)
    #         min_val = min(vals)
    #         if f >= len(self.features):
    #             self.features.append([None, None])
    #         min_f = self.features[f][0]
    #         max_f = self.features[f][1]
    #         if min_f is None or min_f > min_val:
    #             self.features[f][0] = min_val
    #         if max_f is None or max_f < max_val:
    #             self.features[f][1] = max_val
    #         print(self.features[f])
    
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
    
    def add_class(self, c):
        if c not in self.classes:
            self.classes.append(c)
            self.classes.sort()

    def get_normed_feature_vec(self, f):
        normed_f = []
        for a0 in range(f.shape[0]):
            if a0 >= len(normed_f):
                normed_f.append([])
            for a1 in range(f.shape[1]):
                f_range = self.S_features[a0][a1]
                width = (f_range[1] - f_range[0]) if (f_range[1] - f_range[0]) != 0 else max(f_range[1], 1) / 10
                normed = (f_range[1] - f[a0][a1]) / width
                normed_f[a0].append(normed)
        return normed_f
    
    def get_normed_vec(self, f, f_type = None):
        if f_type is None:
            f_type = "U" if len(f) < len(self.S) else "S"
        
        normalizer = self.S if f_type == "S" else self.U

        normed_f = []
        for i, feature in enumerate(f):
            f_range = normalizer[i]
            width = (f_range[1] - f_range[0]) if (f_range[1] - f_range[0]) != 0 else max(f_range[1], 1) / 10
            normed = (f_range[1] - feature) / width
            normed_f.append(normed)
        return normed_f


def get_metafeature_name(options):
    metafeatures = options['meta-features']

    kv_list = [(k, v) for k, v in metafeatures.items()]

    name = '-'.join([f"{k:3}${v}" for k, v in kv_list])
    return name


def get_cosine_distance(A, B, normalizer, vec_type):
    if not normalizer is None:
        nA = normalizer.get_normed_vec(A, f_type=vec_type)
        nB = normalizer.get_normed_vec(B, f_type=vec_type)
    else:
        nA = A
        nB = B
    c = cosine(nA, nB) if (np.any(nA) and np.any(nB)) else 1 if ((not np.any(nA)) and (not np.any(nB))) else 0
    if np.isnan(c):
        print('error')
    return c


def get_pearson_distance(A, B, normalizer, vec_type):
    if not normalizer is None:
        nA = normalizer.get_normed_vec(A, f_type=vec_type)
        nB = normalizer.get_normed_vec(B, f_type=vec_type)
    else:
        nA = A
        nB = B
    nA_is_constant = np.all(nA == nA[0])
    nB_is_constant = np.all(nB == nB[0])

    p = pearsonr(nA, nB) if (not nA_is_constant and not nB_is_constant) else [1] if (nA_is_constant and nB_is_constant) else [0]
    if np.isnan(p[0]):
        print('error')
    # return 1 - pearsonr(normalizer.get_normed_vec(A, f_type = vec_type), normalizer.get_normed_vec(B, f_type= vec_type))[0]
    return 1 - p[0]


def get_feature_distance(A, B, normalizer, vec_type, dist_func = cosine):
    num_features, num_sources = A.shape
    avg_distance = 0
    A = normalizer.get_normed_feature_vec(A)
    B = normalizer.get_normed_feature_vec(B)
    for f in range(num_features):
        A_feature = A[f]
        B_feature = B[f]

        distance = dist_func(A_feature, B_feature)
        avg_distance += distance
    
    return avg_distance / num_features

def get_source_distance(A, B, normalizer, vec_type, dist_func = cosine):
    # print(A)
    # print(A.shape)
    A = np.transpose(normalizer.get_normed_feature_vec(A))
    B = np.transpose(normalizer.get_normed_feature_vec(B))
    # print(A)
    # print(A.shape)
    # exit()
    num_sources, num_features = A.shape
    avg_distance = 0
    for s in range(num_sources):
        A_source = A[s]
        B_source = B[s]

        distance = dist_func(A_source, B_source)
        if np.isnan(distance):
            print('error')
        avg_distance += distance
        # print(avg_distance)
    if np.isnan(avg_distance / num_sources):
        print('error')
    return avg_distance / num_sources


def get_distance(A, B, normalizer, vec_type, distance_metric = 'cosine'):

    if distance_metric == 'cosine':
        return get_cosine_distance(A, B, normalizer, vec_type)
    if distance_metric == 'pearson':
        return get_pearson_distance(A, B, normalizer, vec_type)
    if distance_metric == 'featurec':
        return get_feature_distance(A, B, normalizer, vec_type, dist_func= lambda A, B: get_cosine_distance(A, B, None, vec_type))
    if distance_metric == 'sourcec':
        return get_source_distance(A, B, normalizer, vec_type, dist_func= lambda A, B: get_cosine_distance(A, B, None, vec_type))
    if distance_metric == 'featurep':
        return get_feature_distance(A, B, normalizer, vec_type, dist_func= lambda A, B: get_pearson_distance(A, B, None, vec_type))
    if distance_metric == 'sourcep':
        return get_source_distance(A, B, normalizer, vec_type, dist_func= lambda A, B: get_pearson_distance(A, B, None, vec_type))



def train_concept_windows(concepts, normalizer, window_size = None, options = None):
    train_len = 10000
    if window_size is None:
        window_size = options['window_size']
    output_path = pathlib.Path.cwd() / "storage" / options['data_name']
    if not output_path.exists():
        output_path.mkdir()

    windows = []

    normalizer_path = output_path / f"{train_len}_normalizer"
    for rep in range(10):
        for concept_gen, concept_name in concepts:
            store_path = output_path / f"{concept_name}_{train_len}_{rep}"
            windows.append([[], concept_name])
                                    
            model = HoeffdingTree()                                                                                                                                                                                                                                                                                                                                                                                                                                                              
            for i in range(train_len):                                                       
                if concept_gen.n_remaining_samples() == 0:
                    concept_gen.restart()
                # print(concept_gen.n_remaining_samples())  
                X,y = concept_gen.next_sample()
                normalizer.add_class(y[0])
                model.partial_fit(X, y, normalizer.classes)



            for i in range(window_size):
                # print(concept_gen.n_remaining_samples())
                if concept_gen.n_remaining_samples() == 0:
                    concept_gen.restart()
                X,y = concept_gen.next_sample()
                p = model.predict(X)
                # print(X)
                # print(y)         
                # print(p)                           
                e = p[0] == y[0]               
                windows[-1][0].append((X[0],y[0], p[0], e))                                                                                                                                               
                windows[-1].append(model)                                                                     
                                                                                    
                with store_path.open('wb') as f:
                    pickle.dump(windows[-1], f)                         
    with normalizer_path.open('wb') as f:
        pickle.dump(normalizer, f)

    return windows, normalizer

def trained_concept_window_meta_info(concepts, extractors, normalizer, window_size = None, options = None, rep = -1):
    train_len = 10000
    if window_size is None:
        window_size = options['window_size']
    output_path = pathlib.Path.cwd() / "storage" / options['data_name']
    if not output_path.exists():
        output_path.mkdir()

    windows = []
    normalizer_path = output_path / f"{train_len}_normalizer"
    for concept_gen, concept_name in concepts:
        store_path = output_path / f"{concept_name}_{train_len}_{rep}"
        # if store_path.exists() and normalizer_path.exists():
        print("loaded")
        with store_path.open('rb') as f:
            window = pickle.load(f)
            windows.append(window)
            gen_item = False
        with normalizer_path.open('rb') as f:
            normalizer = pickle.load(f)

            # for i in range(1000):
            #     if concept_gen.n_remaining_samples() == 0:
            #         concept_gen.restart()
            #     X,y = concept_gen.next_sample()
            #     normalizer.add_class(y[0])
        # else:
        #     print("new")
        #     windows.append([[], concept_name])
                                       
        #     model = HoeffdingTree()                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        #     for i in range(train_len):                                                       
        #         if concept_gen.n_remaining_samples() == 0:
        #             concept_gen.restart()
        #         # print(concept_gen.n_remaining_samples())  
        #         X,y = concept_gen.next_sample()
        #         normalizer.add_class(y[0])
        #         model.partial_fit(X, y, normalizer.classes)



        #     for i in range(window_size):
        #         # print(concept_gen.n_remaining_samples())
        #         if concept_gen.n_remaining_samples() == 0:
        #             concept_gen.restart()
        #         X,y = concept_gen.next_sample()
        #         p = model.predict(X)
        #         # print(X)
        #         # print(y)         
        #         # print(p)                           
        #         e = p[0] == y[0]               
        #         windows[-1][0].append((X[0],y[0], p[0], e))                                                                                                                                               
        #         windows[-1].append(model)                                                                     
                                                                                       
        #         with store_path.open('wb') as f:
        #             pickle.dump(windows[-1], f)                         
        #         with normalizer_path.open('wb') as f:
        #             pickle.dump(normalizer, f)

    return windows, normalizer

def get_concept_stats(timeseries, model, normalizer, options = None):
    concept_stats = {}
    s_fingerprint_vector = []
    u_fingerprint_vector = []
    s_feature_vectors = None
    u_feature_vectors = None


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

    if options['meta-features']['FI']:
        explainer = shap.TreeExplainer(model, X)
        shaps = explainer.shap_values(X)
        mean_shaps = np.mean(np.abs(shaps[0]), axis = 0)
        SHAP_vals = [abs(x) for x in mean_shaps]
    else:
        SHAP_vals = [0 for x in range(X.shape[1])]

    for f1, f in enumerate(features):
        stats = get_timeseries_stats(f, SHAP_vals[f1], options = options)
        concept_stats[f"f{f1}"] = stats
        s_fingerprint_vector = [*s_fingerprint_vector, *list(stats.values())]
        u_fingerprint_vector = [*u_fingerprint_vector, *list(stats.values())]
        if s_feature_vectors is None:
            s_feature_vectors = np.array(list(stats.values())).reshape((-1, 1))
            u_feature_vectors = np.array(list(stats.values())).reshape((-1, 1))
        else:
            s_feature_vectors = np.append(s_feature_vectors, np.array(list(stats.values())).reshape((-1, 1)), axis = 1)
            u_feature_vectors = np.append(u_feature_vectors, np.array(list(stats.values())).reshape((-1, 1)), axis = 1)

    stats = get_timeseries_stats(list(map(float, labels)), options = options)
    concept_stats["labels"] = stats
    s_fingerprint_vector = [*s_fingerprint_vector, *list(stats.values())]
    s_feature_vectors = np.append(s_feature_vectors, np.array(list(stats.values())).reshape((-1, 1)), axis = 1)
    # u_feature_vectors = np.append(u_feature_vectors, np.array(list(stats.values())).reshape((-1, 1)), axis = 1)
    stats = get_timeseries_stats(list(map(float, predictions)), options = options)
    concept_stats["predictions"] = stats
    s_fingerprint_vector = [*s_fingerprint_vector, *list(stats.values())]
    u_fingerprint_vector = [*u_fingerprint_vector, *list(stats.values())]
    s_feature_vectors = np.append(s_feature_vectors, np.array(list(stats.values())).reshape((-1, 1)), axis = 1)
    u_feature_vectors = np.append(u_feature_vectors, np.array(list(stats.values())).reshape((-1, 1)), axis = 1)
    stats = get_timeseries_stats(list(map(float, errors)), options = options)
    concept_stats["errors"] = stats
    s_fingerprint_vector = [*s_fingerprint_vector, *list(stats.values())]
    s_feature_vectors = np.append(s_feature_vectors, np.array(list(stats.values())).reshape((-1, 1)), axis = 1)
    # u_feature_vectors = np.append(u_feature_vectors, np.array(list(stats.values())).reshape((-1, 1)), axis = 1)
    stats = get_timeseries_stats(list(map(float, error_distances)), options = options)
    concept_stats["error_distances"] = stats
    s_fingerprint_vector = [*s_fingerprint_vector, *list(stats.values())]
    s_feature_vectors = np.append(s_feature_vectors, np.array(list(stats.values())).reshape((-1, 1)), axis = 1)
    # u_feature_vectors = np.append(u_feature_vectors, np.array(list(stats.values())).reshape((-1, 1)), axis = 1)
    return concept_stats, s_fingerprint_vector, u_fingerprint_vector, s_feature_vectors, u_feature_vectors

def turningpoints(lst):
    dx = np.diff(lst)
    return np.sum(dx[1:] * dx[:-1] < 0)


def get_timeseries_stats(timeseries, FI = None, options = None):
    # print(len(timeseries))
    stats = {}
    if len(timeseries) < 3:
        stats = {}
        stats["mean"] = 0.1
        stats["stdev"] = 0
        if options['meta-features']['skew']:
            stats["skew"] = 0
        if options['meta-features']['kurtosis']:
            stats["kurtosis"] = 0
        if options['meta-features']['turning_point_rate']:
            stats["turning_point_rate"] = 0
        if options['meta-features']['acf']:
            stats["acf_1"] = 0
            stats["acf_2"] = 0
        if options['meta-features']['pacf']:
            stats["pacf_1"] = 0
            stats["pacf_2"] = 0
        if options['meta-features']['MI']:
            stats["MI"] = 0
        if options['meta-features']['FI']:
            stats["FI"] = 0
        if options['meta-features']['IMF']:
            stats["IMF_0"] = 0
            stats["IMF_1"] = 0
            stats["IMF_2"] = 0
        return stats
        # {
        #     "mean": 0,
        #     "stdev": 0,
        #     "skew": 0,
        #     "kurtosis": 0,
        #     "turning_point_rate": 0,
        #     "acf_1": 0,
        #     "acf_2": 0,
        #     "pacf_1": 0,
        #     "pacf_2": 0,
        #     "MI": 0,
        #     "FI": 0,
        #     "IMF_0": 0,
        #     "IMF_1": 0,
        #     "IMF_2": 0,
        # }
    
    if options['meta-features']['IMF']:
        emd = EMD()
        IMFs = emd(np.array(timeseries), max_imf  = 2)
        for i, imf in enumerate(IMFs):
            stats[f"IMF_{i}"] = perm_entropy(imf)
        for i in range(3):
            if f"IMF_{i}" not in stats:
                stats[f"IMF_{i}"] = 0
    stats["mean"] = statistics.mean(timeseries)
    stats["stdev"] = statistics.stdev(timeseries)
    if options['meta-features']['skew']:
        stats["skew"] = scipy.stats.skew(timeseries)
    if options['meta-features']['kurtosis']:
        stats['kurtosis'] = scipy.stats.kurtosis(timeseries)
    if options['meta-features']['turning_point_rate']:
        tp = int(turningpoints(timeseries))
        tp_rate = tp / len(timeseries)
        stats['turning_point_rate'] = tp_rate
    
    if options['meta-features']['acf']:
        try:
            acf_vals = acf(timeseries, nlags = 3, fft=True)
        except Exception as e:
            print(e)
            exit()
            acf_vals = [-1 for x in range(6)]
        for i,v in enumerate(acf_vals):
            if i == 0: continue
            if i > 2: break
            stats[f"acf_{i}"] = v if not np.isnan(v) else -1
    
    if options['meta-features']['pacf']:
        try:
            acf_vals = pacf(timeseries, nlags = 3)
        except Exception as e:
            print(e)
            acf_vals = [-1 for x in range(6)]
        for i,v in enumerate(acf_vals):
            if i == 0: continue
            if i > 2: break
            stats[f"pacf_{i}"] = v if not np.isnan(v) else -1

    if options['meta-features']['MI']:
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
    
    if options['meta-features']['FI']:
        stats["FI"] = FI if FI is not None else 0

    # print(stats)
    return stats
def window_to_timeseries(window):
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


def learn_fingerprints_from_concepts(concepts, normalizer, options):
    for concept_gen, _ in concepts:
        concept_gen.prepare_for_use()
    fingerprints = []
    stats = {}
    SFs = {}
    UFs = {}
    SFFs = {}
    UFFs = {}
    for i in tqdm.tqdm(range(10)):
        windows, normalizer = trained_concept_window_meta_info(concepts, None, normalizer, options = options, rep=i)
        for w in windows:
            # print(w[1])
            if w[1] not in stats:
                stats[w[1]] = {}
            if w[1] not in SFs:
                SFs[w[1]] = []
            if w[1] not in UFs:
                UFs[w[1]] = []
            if w[1] not in SFFs:
                SFFs[w[1]] = []
            if w[1] not in UFFs:
                UFFs[w[1]] = []
            timeseries = window_to_timeseries(w)
            print("getting stats")
            concept_stats, sfingerprint, ufingerprint, sfeaturefinger, ufeaturefinger = get_concept_stats(timeseries, w[2], normalizer, options = options)
            print("got stats")
            normalizer.add_observation(sfingerprint, ufingerprint)
            normalizer.add_feature_observations(sfeaturefinger, ufeaturefinger)
            # print(concept_stats)
            # print(sfingerprint)
            # print(ufingerprint)
            SFs[w[1]].append(sfingerprint)
            UFs[w[1]].append(ufingerprint)
            SFFs[w[1]].append(sfeaturefinger)
            UFFs[w[1]].append(ufeaturefinger)

            for k in concept_stats.keys():
                if k not in stats[w[1]]:
                    stats[w[1]][k] = {}
                for s in concept_stats[k].keys():
                    if s not in stats[w[1]][k]:
                        stats[w[1]][k][s]= []
                    stats[w[1]][k][s].append(concept_stats[k][s])


    for concept in SFs.keys():
        fingerprints.append((np.mean(SFs[concept], axis = 0), np.mean(UFs[concept], axis = 0), np.mean(SFFs[concept], axis = 0), np.mean(UFFs[concept], axis = 0), concept))

    return fingerprints, stats, normalizer


def get_sliding_window_similarities(stream, length, fingerprints, normalizer, options = None):
    window_size = options['window_size']
    observation_gap = options['observation_gap']
    detector = ADWIN()
    classifier = HoeffdingTree()
    stats = []
    sims = {}
    window = deque(maxlen=window_size)
    stream.prepare_for_use()
    last_stats = None
    last_sims = None
    right = 0
    wrong = 0
    for i in tqdm.tqdm(range(length - 1)):
    # for i in range(length - 1):
        # print(i)
        X,y = stream.next_sample()
        p = classifier.predict(X)
        e = y[0] == p[0]
        right += y[0] == p[0]
        wrong += y[0] != p[0]
        classifier.partial_fit(X, y, normalizer.classes)

        detector.add_element(e)
        if detector.detected_change():
            detector = ADWIN()
            classifier = HoeffdingTree()
        
        window.append((X[0], y[0], p[0], e))

        if i > window_size:
            if i % observation_gap == 0:
                current_timeseries = window_to_timeseries([window, "n"])
                concept_stats, sfingerprint, ufingerprint, sfeaturefinger, ufeaturefinger = get_concept_stats(current_timeseries, classifier, normalizer, options = options)
                normalizer.add_observation(sfingerprint, ufingerprint)
                normalizer.add_feature_observations(sfeaturefinger, ufeaturefinger)
                stats.append((concept_stats, i))
                for sf, uf, sff, uff, n in fingerprints:
                    # sim = cosine(normalizer.get_normed_vec(sf), normalizer.get_normed_vec(sfingerprint))
                    # sSim = 1 - pearsonr(normalizer.get_normed_vec(sf, f_type = "S"), normalizer.get_normed_vec(sfingerprint, f_type= "S"))[0]
                    # uSim = 1 - pearsonr(normalizer.get_normed_vec(uf, f_type = "U"), normalizer.get_normed_vec(ufingerprint, f_type= "U"))[0]
                    if n not in sims:
                        sims[n] = []
                    sSim_cosine = get_distance(sf, sfingerprint, normalizer, "S")
                    sSim_pearson = get_distance(sf, sfingerprint, normalizer, "S", "pearson")
                    uSim_cosine = get_distance(uf, ufingerprint, normalizer, "U")
                    uSim_pearson = get_distance(uf, ufingerprint, normalizer, "U", "pearson")

                    csfeature_distance = get_distance(sff, sfeaturefinger, normalizer, "S", "featurec")
                    cufeature_distance = get_distance(uff, ufeaturefinger, normalizer, "S", "featurec")
                    cssource_distance = get_distance(sff, sfeaturefinger, normalizer, "S", "sourcec")
                    cusource_distance = get_distance(uff, ufeaturefinger, normalizer, "S", "sourcec")
                    psfeature_distance = get_distance(sff, sfeaturefinger, normalizer, "S", "featurep")
                    pufeature_distance = get_distance(uff, ufeaturefinger, normalizer, "S", "featurep")
                    pssource_distance = get_distance(sff, sfeaturefinger, normalizer, "S", "sourcep")
                    pusource_distance = get_distance(uff, ufeaturefinger, normalizer, "S", "sourcep")
                    # sims[n].append((sSim_cosine, uSim_cosine, sSim_pearson, uSim_pearson, sfeature_distance, ufeature_distance, ssource_distance, usource_distance,  i))
                    similarities = {'s_cosign': sSim_cosine,
                                    'u_cosine': uSim_cosine,
                                    's_pearson': sSim_pearson,
                                    'u_pearson': uSim_pearson,
                                    'cs_by_feature': csfeature_distance,
                                    'cu_by_feature': cufeature_distance, 
                                    'cs_by_source': cssource_distance,
                                    'cu_by_source': cusource_distance,
                                    'ps_by_feature': psfeature_distance,
                                    'pu_by_feature': pufeature_distance, 
                                    'ps_by_source': pssource_distance,
                                    'pu_by_source': pusource_distance}
                    # print(similarities)
                    sims[n].append((similarities,  i))

    print(f"Accuracy: {right / (right + wrong)}")
    return stats, sims, right, wrong

if __name__ == "__main__":
    #%%

    # name = "stagger"
    # data_type = "synthetic"
    # data_source = "STAGGER"
    name = "qg"
    data_type = "real"
    data_source = "Real"

    if data_type == "synthetic":
        concepts = load_synthetic_concepts(name, data_source, 0)
    else:
        concepts = load_real_concepts(name, data_source, 0)

    # concepts = load_synthetic_concepts("test", "STAGGER", 0)
    # concepts = load_real_concepts("UCI-Wine", "Real", 0)
    # concepts = load_synthetic_concepts("testARGWAL", "ARGWAL", 0)
    # concepts = load_synthetic_concepts("testTREE", "RTREE", 0)
    normalizer = Normalizer()
    # concepts = load_synthetic_concepts("testSEA", "SEA", 0)

    fingerprints, fingerprint_stats = learn_fingerprints_from_concepts(concepts, normalizer)

    #%%


    if data_type == "synthetic":
        concepts = load_synthetic_concepts(name, data_source, 0)
    else:
        concepts = load_real_concepts(name, data_source, 0)
    # concepts = load_synthetic_concepts("test", "STAGGER", 0)
    # concepts = load_real_concepts("UCI-Wine", "Real", 0)
    stream_concepts, length = get_inorder_concept_ranges(concepts)
    stream = AbruptDriftStream(stream_concepts, length)
    # stream, stream_concepts, length = stitch_concepts_inorder(concepts)

    stats, sims, right, wrong = get_sliding_window_similarities(stream, length, fingerprints, normalizer)

    output_path = pathlib.Path.cwd() / "output" / name

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
