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
obs
#%%
df = pd.DataFrame(obs)
df.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9' ][:len(df.columns)]
# %%
df = df.reset_index()
df
# %%
fig, ax = plt.subplots()
val = 'f9'
sns.scatterplot(data=df, x='index', y=val, ax=ax)
sns.scatterplot( x=df['index'], y=df[val].rolling(75).mean(), ax=ax)
plt.plot()
# %%

# %%

# %%
