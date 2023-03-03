#%%
import numpy as np
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, lognorm

from ConceptFingerprint.Data.load_data import \
        load_synthetic_concepts,\
        load_real_concepts,\
        get_inorder_concept_ranges,\
        AbruptDriftStream
from ConceptFingerprint.Classifier.feature_selection.fisher_score import fisher_score
from ConceptFingerprint.Classifier.feature_selection.mutual_information import bin_X, MI_histogram_estimation_cache_mat, MI_estimation_cache_mat
from ConceptFingerprint.Classifier.meta_info_classifier import DSClassifier, get_dimension_weights
from ConceptFingerprint.Classifier.hoeffding_tree_shap import HoeffdingTreeSHAPClassifier
sns.set()

#%%

# concepts = load_synthetic_concepts('RTREESAMPLEHP-A',
#                                     102,
#                                     raw_data_path = pathlib.Path(r'S:\PhD\Packages\ConceptFingerprint\RawData') / 'Synthetic')
concepts = load_synthetic_concepts('RTREESAMPLE-NB',
                                    20,
                                    raw_data_path = pathlib.Path(r'S:\PhD\Packages\ConceptFingerprint\RawData') / 'Synthetic')

# concept_gen.prepare_for_use()
stream_concepts, length = get_inorder_concept_ranges(concepts, concept_length=2500, seed=20, repeats=3, concept_max=10, repeat_proportion=1.0, shuffle=False)
stream = AbruptDriftStream(stream_concepts, length)
# stream.prepare_for_use()
all_classes = stream._get_target_values()
# %%
obs = []
for i in range(length):
    X,y = stream.next_sample()
    obs.append(X[0])

stream = AbruptDriftStream(stream_concepts, length)
obs


#%%
df = pd.DataFrame(obs)
df.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9' ][:len(df.columns)]
# %%
df = df.reset_index()
df
# %%
sns.scatterplot(data=df, x='index', y='f5')
# %%

# %%
classifier = DSClassifier(
                learner = HoeffdingTreeSHAPClassifier,
                window_size = 75,
                similarity_gap = 4,
                fingerprint_update_gap = 4,
                non_active_fingerprint_update_gap = 10,
                observation_gap = 50,
                sim_measure = 'metainfo',
                ignore_sources = ['labels', 'predictions', 'error_distances'],
                ignore_features = ['IMF', 'MI', 'FI', 'turning_point_rate', 'acf', 'pacf', 'kurtosis', 'skew', 'stdev'],
                similarity_num_stdevs = 3,
                similarity_min_stdev = 0.005,
                similarity_max_stdev = 0.1,
                buffer_ratio=0.25,
                feature_selection_method='Cachehistogram_MI',
                fingerprint_method='cachehistogram',
                fingerprint_bins=2,
                MI_calc='metainfo')

#%%
for i in range(int(length * 0.40)):
    X,y = stream.next_sample()
    classifier.partial_fit(X, y, classes=all_classes)

#%%
print(X)
classifier.get_similarity_to_active_state(classifier.active_metainfo, flat_nonorm_current_metainfo=classifier.active_nonormed_flat)

#%%
state_non_active_fingerprints = {k: (v.fingerprint, v.non_active_fingerprints) for k,v in classifier.state_repository.items() if v.fingerprint is not None}
weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = get_dimension_weights(list([state.fingerprint for state in classifier.state_repository.values() if state.fingerprint is not None]), state_non_active_fingerprints,  classifier.normalizer, feature_selection_method=classifier.feature_selection_method)
# %%
