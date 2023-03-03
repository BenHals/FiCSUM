#%%
import pathlib
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#%%
results_directory = pathlib.Path(r"H:\PhD\FingerprintExtension\results\test1")
results_directory = pathlib.Path(r"H:\PhD\PythonPackages\ConceptFingerprint\output\expDefault")
result_files = results_directory.rglob('results_run*')
results = []
for rf in result_files:
    results.append(json.load(rf.open('r')))
overall_df = pd.DataFrame(results)
overall_df
#%%
df
#%%
# df = overall_df[overall_df['data_name'] == "AQSex"]
df = overall_df[overall_df['data_name'] == "cmc"]
# %%
df.columns
# %%

# histogram_df = df[df['fs_method'] == 'Cachehistogram_MI']
histogram_df = df[df['fs_method'] == 'CacheMIHy']
histogram_df.groupby(('fingerprint_bins')).aggregate(('mean', 'std'))[['overall_accuracy', 'kappa', 'GT_mean_f1', 'overall_time', 'overall_mem', 'peak_fingerprint_mem', 'average_fingerprint_mem']]
# %%
sns.lineplot(data=histogram_df, x='fingerprint_bins', y='overall_accuracy')
# %%
