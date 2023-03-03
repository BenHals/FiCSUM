
#%%
from ConceptFingerprint.Classifier.feature_selection.mutual_information import information_gain_normal_distributions_HxCond_UN,information_gain_normal_distributions_UN,information_gain_normal_distributions_HxCond, information_gain_normal_distributions_Hx,normal_distribution_entropy,MI_estimation, information_gain_normal_distributions_JS, information_gain_normal_distributions_uniform, information_gain_normal_distributions, information_gain_normal_distributions_KL, information_gain_normal_distributions_swap, information_gain_normal_distributions_sym, KL_divergence
import numpy as np
import pandas as pd
import math
import random
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import tqdm
from ConceptFingerprint.Classifier.meta_info_classifier import (
    get_dimension_weights
)
from ConceptFingerprint.Classifier.meta_info_classifier import DSClassifier
from ConceptFingerprint.Classifier.hoeffding_tree_shap import HoeffdingTreeSHAPClassifier
import matplotlib.pyplot as plt

def next_sample(concept):
    stats = {}
    X_vec = []
    if concept == 0:
        source1 = {}
        source1["F1"] = np.random.normal(0, 1)
        source1["F2"] = np.random.random()
        source1["F3"] = np.random.random() * 10
        stats["S1"] = source1
        source2 = {}
        source2["F1"] = np.random.random()
        source2["F2"] = np.random.random() * 5
        source2["F3"] = np.random.random() * 10
        stats["S2"] = source2
        source3 = {}
        source3["F1"] = np.random.normal(0, 1)
        source3["F2"] = np.random.normal(5, 1)
        source3["F3"] = np.random.normal(10, 5)
        stats["S3"] = source3
    else:
        source1 = {}
        source1["F1"] = np.random.normal(0, 1)
        source1["F2"] = np.random.random()
        source1["F3"] = np.random.random() * 10
        stats["S1"] = source1
        source2 = {}
        source2["F1"] = np.random.random() + 10
        source2["F2"] = np.random.random()
        source2["F3"] = np.random.random() * 5 + 5
        stats["S2"] = source2
        source3 = {}
        source3["F1"] = np.random.normal(10, 1)
        source3["F2"] = np.random.normal(5, 5)
        source3["F3"] = np.random.normal(0, 1)
        stats["S3"] = source3
    for s in stats:
        for f in stats[s]:
            X_vec.append(stats[s][f])
    y = 0 if sum(X_vec) < 25 else 1
    if concept > 0:
        y = 1 if sum(X_vec) < 25 else 0
    return stats, X_vec, y
def next_sample_small(concept):
    stats = {}
    X_vec = []
    if concept == 0:
        source1 = {}
        #f0
        source1["F1"] = np.random.normal(0, 1)
        #f1
        source1["F2"] = np.random.random()
        stats["S1"] = source1
        source2 = {}
        #f2
        source2["F1"] = np.random.normal(0, 1)
        #f3
        source2["F2"] = np.random.random()
        stats["S2"] = source2
    else:
        source1 = {}
        #f0
        source1["F1"] = np.random.normal(0, 1)
        #f1
        source1["F2"] = np.random.random()
        stats["S1"] = source1
        source2 = {}
        #f2
        source2["F1"] = np.random.normal(10, 1)
        #f3
        source2["F2"] = np.random.random() * 5 + 5
        stats["S2"] = source2
    for s in stats:
        for f in stats[s]:
            X_vec.append(stats[s][f])
    y = 0 if sum(X_vec) < 25 else 1
    if concept > 0:
        y = 1 if sum(X_vec) < 25 else 0
    return stats, X_vec, y

#%%
observations = []
for i in range(5000):
    stats, X, y = next_sample_small(0 if i < 2500 else 1)
    observations.append((X, y))

#%%
stats, X, y = next_sample_small(0 if i < 2500 else 1)
plt.clf()
for f in range(len(X)):
    plt.plot([x[f] for x,y in observations])
plt.show()


# def test_integration():
#%%
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='fisher')
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='default')
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='histogram_MI', fingerprint_method = "histogram", fingerprint_bins=15)
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='MIHy', fingerprint_method = "histogram", fingerprint_bins=15)
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='MIHxCond', fingerprint_method = "histogram", fingerprint_bins=15)
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='sketch_MI', fingerprint_method = "cachesketch", fingerprint_bins=100)
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='CacheMIHy', fingerprint_method = "cache", fingerprint_bins=100)
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='sketch_covMI', fingerprint_method = "cachesketch", fingerprint_bins=100)
fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='sketch_covredMI', fingerprint_method = "cachesketch", fingerprint_bins=100)

monitor = []
force = True
for i, (X, y) in enumerate(observations):
    # if i == 3240:
    #     break
    observation_monitoring = {}
    drift_occured = False
    concept_drift = False
    concept_drift_target = None
    concept_transition = False
    fisher_cls.manual_control = False
    fisher_cls.force_stop_learn_fingerprint = False
    fisher_cls.force_transition = False
    window_size = 75
    for c in [1]:
        concept_start= 2500
        if i == concept_start + window_size + 50:
        # if i == concept_start:
            concept_drift = True
            concept_drift_target = c
    if concept_drift and force:
        fisher_cls.manual_control = True
        fisher_cls.force_transition = True
        fisher_cls.force_stop_learn_fingerprint = True
        fisher_cls.force_transition_to = concept_drift_target
    if force:
        fisher_cls.force_transition_only = True
    fisher_cls.partial_fit([X], [y], classes=[0, 1])
    current_active_model = fisher_cls.active_state
    observation_monitoring['active_model'] = current_active_model
    weights = fisher_cls.monitor_feature_selection_weights
    if weights is not None:
        observation_monitoring['feature_weights'] = {}
        for s,f,v in weights:
            if s not in observation_monitoring['feature_weights']:
                observation_monitoring['feature_weights'][s] = {}
            observation_monitoring['feature_weights'][s][f] = v
    else:
        observation_monitoring['feature_weights'] = None
    monitor.append(observation_monitoring)
#%%
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='fisher')
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='default')
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='histogram_MI', fingerprint_method = "histogram", fingerprint_bins=15)
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='MIHy', fingerprint_method = "cache", fingerprint_bins=10)
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='CacheMIHy', fingerprint_method = "cache", fingerprint_bins=10)
fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='CacheMIHy', fingerprint_method = "cache", fingerprint_bins=10)
fisher_cls2 = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='sketch_MI', fingerprint_method = "cachesketch", fingerprint_bins=10)

# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='histogram_MI', fingerprint_method = "cachehistogram", fingerprint_bins=10)
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='oldhistogram_MI', fingerprint_method = "cachehistogram", fingerprint_bins=10)
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='Cachehistogram_MI', fingerprint_method = "cachehistogram", fingerprint_bins=10)
# fisher_cls2 = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='Cachehistogram_MI', fingerprint_method = "cachehistogram", fingerprint_bins=50)
# fisher_cls2 = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='histogram_MI', fingerprint_method = "cachehistogram", fingerprint_bins=10)
# fisher_cls = DSClassifier(learner=HoeffdingTreeSHAPClassifier, feature_selection_method='MIHxCond', fingerprint_method = "histogram", fingerprint_bins=15)
monitor = []
monitor2 = []
force = True
for i, (X, y) in enumerate(observations):
    if i == 3500:
        break
    observation_monitoring = {}
    observation_monitoring2= {}
    drift_occured = False
    concept_drift = False
    concept_drift_target = None
    concept_transition = False
    fisher_cls.manual_control = False
    fisher_cls2.manual_control = False
    fisher_cls.force_stop_learn_fingerprint = False
    fisher_cls2.force_stop_learn_fingerprint = False
    fisher_cls.force_transition = False
    fisher_cls2.force_transition = False
    window_size = 75
    for c in [1]:
        concept_start= 2500
        if i == concept_start + window_size + 10:
        # if i == concept_start:
            concept_drift = True
            concept_drift_target = c
    if concept_drift and force:
        fisher_cls.manual_control = True
        fisher_cls2.manual_control = True
        fisher_cls.force_transition = True
        fisher_cls2.force_transition = True
        fisher_cls.force_stop_learn_fingerprint = True
        fisher_cls2.force_stop_learn_fingerprint = True
        fisher_cls.force_transition_to = concept_drift_target
        fisher_cls2.force_transition_to = concept_drift_target
    if force:
        fisher_cls.force_transition_only = True
        fisher_cls2.force_transition_only = True
    fisher_cls.partial_fit([X], [y], classes=[0, 1])
    fisher_cls2.partial_fit([X], [y], classes=[0, 1])
    current_active_model = fisher_cls.active_state
    current_active_model2 = fisher_cls2.active_state
    observation_monitoring['active_model'] = current_active_model
    observation_monitoring2['active_model'] = current_active_model2
    weights = fisher_cls.monitor_feature_selection_weights
    weights2 = fisher_cls2.monitor_feature_selection_weights
    if weights is not None:
        observation_monitoring['feature_weights'] = {}
        for s,f,v in weights:
            if s not in observation_monitoring['feature_weights']:
                observation_monitoring['feature_weights'][s] = {}
            observation_monitoring['feature_weights'][s][f] = v
    else:
        observation_monitoring['feature_weights'] = None
    if weights2 is not None:
        observation_monitoring2['feature_weights'] = {}
        for s,f,v in weights2:
            if s not in observation_monitoring2['feature_weights']:
                observation_monitoring2['feature_weights'][s] = {}
            observation_monitoring2['feature_weights'][s][f] = v
    else:
        observation_monitoring2['feature_weights'] = None
    monitor.append(observation_monitoring)
    monitor2.append(observation_monitoring2)
#%%
print("hello")
#%%
print([v['active_model'] for v in monitor])
#%%
feature_weights = {}
for s in fisher_cls.normalizer.source_order:
    feature_weights[s] = {}
    for f in fisher_cls.normalizer.feature_order:
        feature_weights[s][f] = [x['feature_weights'][s][f] if x is not None and s in x['feature_weights'] else -1 for x in monitor]
        if s in ['f0', 'f1', 'f2', 'f3']:
            if f in ['mean', 'stdev']:
                plt.plot(feature_weights[s][f], label=f"{s}-{f}")
# print(feature_weights)
plt.legend()
plt.show()
#%%
feature_weights2 = {}
for s in fisher_cls2.normalizer.source_order:
    feature_weights2[s] = {}
    for f in fisher_cls2.normalizer.feature_order:
        feature_weights2[s][f] = [x['feature_weights'][s][f] if x is not None and s in x['feature_weights'] else -1 for x in monitor2]
        if s in ['f0', 'f1', 'f2', 'f3']:
            if f in ['mean', 'stdev']:
                plt.plot(feature_weights2[s][f], label=f"{s}-{f}")
# print(feature_weights)
plt.legend()
plt.show()

# test_integration()
# %%
stats, X, y = next_sample_small(0)
stats
# %%
fisher_cls.states
# %%
state_non_active_fingerprints = {k: (v.fingerprint, v.non_active_fingerprints) for k,v in fisher_cls.state_repository.items() if v.fingerprint is not None}
weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = get_dimension_weights(list([state.fingerprint for state in fisher_cls.state_repository.values() if state.fingerprint is not None]), state_non_active_fingerprints,  fisher_cls.normalizer, feature_selection_method=fisher_cls.feature_selection_method)
# weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = get_dimension_weights(list([state.fingerprint for state in fisher_cls2.state_repository.values() if state.fingerprint is not None]), state_non_active_fingerprints,  fisher_cls2.normalizer, feature_selection_method=fisher_cls2.feature_selection_method)
# %%
weights
# %%
fisher_cls.states[0].fingerprint.fingerprint['f3']