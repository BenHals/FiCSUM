#%%
from ConceptFingerprint.Classifier.feature_selection.mutual_information import information_gain_normal_distributions_HxCond_UN,information_gain_normal_distributions_UN,information_gain_normal_distributions_HxCond, information_gain_normal_distributions_Hx,normal_distribution_entropy,MI_estimation, information_gain_normal_distributions_JS, information_gain_normal_distributions_uniform, information_gain_normal_distributions, information_gain_normal_distributions_KL, information_gain_normal_distributions_swap, information_gain_normal_distributions_sym, KL_divergence
import numpy as np
import pandas as pd
import math
import random
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import tqdm
from ConceptFingerprint.Classifier.fingerprint import (
    FingerprintBinning,
)
from ConceptFingerprint.Classifier.feature_selection.online_feature_selection import mi_from_fingerprint_histogram
from ConceptFingerprint.Classifier.normalizer import Normalizer
#%%
observations = []
for i in tqdm.tqdm(range(50)):
    mu_A = random.random() * 100
    sigma_A = random.random() * 20
    mu_B = random.random() * 100
    sigma_B = random.random() * 20
    mu_C = random.random() * 100
    sigma_C = random.random() * 20
    mu_D = random.random() * 100
    sigma_D = random.random() * 20
    dist_1 = np.random.normal(mu_A, sigma_A, 100000)
    y_1 = np.array([0 for i in range(100000)])
    dist_2 = np.random.normal(mu_B, sigma_B, 100000)
    y_2 = np.array([1 for i in range(100000)])
    dist_3 = np.random.normal(mu_C, sigma_C, 100000)
    y_3 = np.array([2 for i in range(100000)])
    dist_4 = np.random.normal(mu_D, sigma_D, 100000)
    y_4 = np.array([3 for i in range(100000)])

    dists = [dist_1, dist_2, dist_3, dist_4]
    ys = (y_1, y_2, y_3, y_4)
    mus = [mu_A, mu_B, mu_C, mu_D]
    sigmas = [sigma_A, sigma_B, sigma_C, sigma_D]
    dists = [dist_1, dist_2, dist_3]
    ys = (y_1, y_2, y_3)
    mus = [mu_A, mu_B, mu_C]
    sigmas = [sigma_A, sigma_B, sigma_C]
    all_X = np.concatenate(dists)
    all_y = np.concatenate(ys)

    def np_construct(*args, **kwargs):
        kwargs['num_bins'] = 500
        return FingerprintBinning(*args, **kwargs)
    norm = Normalizer(fingerprint_constructor = np_construct)
    concepts = []
    for d in dists:
        fp = None
        for v in d:
            stats = {"source": {"feature": v}}
            norm.add_stats(stats)
            if fp is None:
                fp = FingerprintBinning(stats, norm, num_bins = 500)
            else:
                fp.incorperate(stats)
        concepts.append(fp)


    # Assuming a normal distribution, what is the reduction in mutual information between X and y?
    information_gain = information_gain_normal_distributions(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 100)
    information_gain_UN = information_gain_normal_distributions_UN(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 100)
    # information_gain_uniform = information_gain_normal_distributions_uniform(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 100)
    # information_gain_js = information_gain_normal_distributions_JS(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 100)
    # information_gain_2 = information_gain_normal_distributions_KL(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 100)
    # information_gain_swap = information_gain_normal_distributions_swap(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 100)
    # information_gain_sym = information_gain_normal_distributions_sym(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 100)
    information_gain_bin_10 = MI_estimation(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists])
    information_gain_bin_100 = MI_estimation(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 100)
    # information_gain_bin_2 = MI_estimation(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 2)
    information_gain_bin_Hx = information_gain_normal_distributions_Hx(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 100)
    information_gain_bin_HxCond = information_gain_normal_distributions_HxCond(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 100)
    information_gain_bin_HxCond_UN = information_gain_normal_distributions_HxCond_UN(np.mean(all_X), np.std(all_X), [np.min(all_X), np.max(all_X)], [np.mean(x) for x in dists], [np.std(x) for x in dists], [len(d) for d in dists], 100)
    skl_IG = mutual_info_classif(all_X.reshape(-1, 1), all_y)[0]
    histogram_MI = mi_from_fingerprint_histogram(norm, concepts, "source", "feature")
    # print(skl_IG)
    # print(information_gain)
    # print(information_gain_2)
    # print(information_gain_swap)
    # print(information_gain_sym)

    observation = {
        'mu_A': mu_A,
        'sigma_A': sigma_A,
        'mu_B': mu_B,
        'sigma_B': sigma_B,
        'mu_O': np.mean(all_X),
        'sigma_O': np.std(all_X),
        'skl_IG': skl_IG,
        'group_difference': np.mean(mus) / np.mean(sigmas),
        'information_gain': information_gain,
        # 'information_gain_uniform': information_gain_uniform,
        # 'information_gain_js': information_gain_js,
        # 'information_gain_2': information_gain_2,
        # 'information_gain_swap': information_gain_swap,
        # 'information_gain_sym': information_gain_sym,
        'information_gain_bin_10': information_gain_bin_10,
        'information_gain_bin_100': information_gain_bin_100,
        # 'information_gain_bin_2': information_gain_bin_2,
        'information_gain_bin_Hx': information_gain_bin_Hx,
        'information_gain_bin_HxCond': information_gain_bin_HxCond,
        "information_gain_UN": information_gain_UN,
        "information_gain_bin_HxCond_UN": information_gain_bin_HxCond_UN,
        "histogram_MI": histogram_MI
    }
    observations.append(observation)

#%%
df = pd.DataFrame(observations)
df['diff'] = df['mu_A'] - df['mu_B']
df['diff_sd'] = abs(df['mu_A'] - df['mu_B']) / (max(df['sigma_A'].max(), df['sigma_B'].max()))
df['information_gain_half'] = df['information_gain'] * 0.55
df['n_skl_IG'] = (df['skl_IG'] - df['skl_IG'].min()) / (df['skl_IG'].max() - df['skl_IG'].min())
df['n_information_gain'] = (df['information_gain'] - df['information_gain'].min()) / (df['information_gain'].max() - df['information_gain'].min())
df['n_information_gain_bin_100'] = (df['information_gain_bin_100'] - df['information_gain_bin_100'].min()) / (df['information_gain_bin_100'].max() - df['information_gain_bin_100'].min())
df['n_information_gain_bin_10'] = (df['information_gain_bin_10'] - df['information_gain_bin_10'].min()) / (df['information_gain_bin_10'].max() - df['information_gain_bin_10'].min())
df['n_information_gain_bin_HxCond'] = (df['information_gain_bin_HxCond'] - df['information_gain_bin_HxCond'].min()) / (df['information_gain_bin_HxCond'].max() - df['information_gain_bin_HxCond'].min())
df['n_information_gain_bin_Hx'] = (df['information_gain_bin_Hx'] - df['information_gain_bin_Hx'].min()) / (df['information_gain_bin_Hx'].max() - df['information_gain_bin_Hx'].min())
df['n_histogram_MI'] = (df['histogram_MI'] - df['histogram_MI'].min()) / (df['histogram_MI'].max() - df['histogram_MI'].min())
df['c_information_gain'] = (df['information_gain'].clip(upper=2))
# df['c_information_gain_swap'] = (df['information_gain_swap'].clip(upper=1))
df.head()
# %%
sns.set()
sns.scatterplot(data=df, x='diff', y='skl_IG')
#%%
sns.scatterplot(data=df, x='diff', y='information_gain_bin_HxCond')
#%%
sns.scatterplot(data=df, x='diff_sd', y='information_gain')
#%%
sns.scatterplot(data=df, x='sigma_B', y='information_gain')
# %%
sns.scatterplot(data=df, x='sigma_O', y='n_skl_IG')
sns.scatterplot(data=df, x='sigma_O', y='n_information_gain')

#%%
sns.scatterplot(data=df, x='diff', y='skl_IG')
sns.scatterplot(data=df, x='diff', y='information_gain')
# %%
sns.scatterplot(data=df, x='n_skl_IG', alpha=0.5, label="skl", y='n_skl_IG')
sns.scatterplot(data=df, x='n_skl_IG', alpha=0.5, label="estimate from Y", y='n_information_gain_bin_100')
sns.scatterplot(data=df, x='n_skl_IG', alpha=0.5, label="estimate from Y 10", y='n_information_gain_bin_10')
sns.scatterplot(data=df, x='n_skl_IG', alpha=0.5, label="estimate from X", y='n_information_gain_bin_HxCond')
sns.scatterplot(data=df, x='n_skl_IG', alpha=0.5, label="estimate initial X", y='n_information_gain_bin_Hx')
sns.scatterplot(data=df, x='n_skl_IG', alpha=0.5, label="use formula for X", y='n_information_gain')
sns.scatterplot(data=df, x='n_skl_IG', alpha=0.5, label="use histogram for X", y='n_histogram_MI')

# %%
sns.scatterplot(data=df, x='information_gain_UN', y='information_gain_UN')
sns.scatterplot(data=df, x='information_gain_UN', y='information_gain_2')

# %%
sns.scatterplot(data=df, x='information_gain_bin_HxCond', y='information_gain_bin_HxCond')
sns.scatterplot(data=df, x='information_gain_bin_HxCond', y='information_gain_bin_HxCond_UN')

# %%
df['histogram_MI']
# %%
