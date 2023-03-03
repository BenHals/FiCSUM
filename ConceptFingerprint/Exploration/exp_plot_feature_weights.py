#%%
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
log_path = pathlib.Path(r'H:\output\expDefault\ConceptFingerprint-master-2dfb6aa\Arabic\1001\run_1229201974012767153_0.csv')
log_path_2 = pathlib.Path(r'H:\output\expDefault\ConceptFingerprint-master-2dfb6aa\Arabic\1001\run_3369313819230199491_0.csv')
log_path_2003_fisher = pathlib.Path(r'H:\output\expDefault\ConceptFingerprint-master-2dfb6aa\Arabic\2003\run_9188542613100145381_0.csv')
log_path_2003_MIM = pathlib.Path(r'H:\output\expDefault\ConceptFingerprint-master-2dfb6aa\Arabic\2003\run_-7177387291253706998_0.csv')
log = pd.read_csv(log_path)
# %%

def plot_features(log):
    feature_weights = log['feature_weights'].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True)
    lt = None
    for c in feature_weights.columns:
        features = feature_weights[c].astype(str).str.split(":", expand=True)
        features['ts'] = features.index
        if lt is None:
            lt = features
        else:
            lt = pd.concat([lt, features], ignore_index=True)
    lt['value'] = lt[1].astype(float)
    lt = lt.sort_values(axis=0, by='ts')
    features = list(lt[0].unique())
    feature_median = []
    for feature in features:
        feature_median.append((feature, lt[lt[0] == feature]['value'].quantile(0.5)))
    print(feature_median[0])
    feature_median.sort(key = lambda x: x[1], reverse=True)

    display_val = lt['value'].quantile(1 - 1/len(features))
    for feature, median in feature_median[:10]:
        # feature_max = lt[lt[0] == feature]['value'].max()
        # lab = feature if feature_max > display_val else None
        lab = feature
        sns.lineplot(x='ts', y='value', label=lab, data=lt[lt[0] == feature])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.plot()
# %%

lt
#%%
# log = pd.read_csv(log_path_2)
log = pd.read_csv(log_path)
plot_features(log)
# %%


# %%


# %%
lt['ts']
# %%
