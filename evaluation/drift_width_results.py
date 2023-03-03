#%%
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
sns.set()
sns.set_style('whitegrid')
sns.set_context('paper')

#%%
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-width")
results_files = data_dir.rglob("result*")
results = []
for rf in results_files:
    result = json.load(rf.open('r'))
    if 'feature_weights' in result:
        for k in result['feature_weights']:
            result[f"fw-{k}"] = result['feature_weights'][k]
    results.append(result)
all_df = pd.DataFrame(results)
# %%
all_df.columns
#%%

# %%
# df = all_df
# df = all_df[(all_df['data_name'] == 'SigNoiseGenerator-4') & (all_df['fs_method'] == 'CacheMIHy')]
df = all_df[(~all_df['optdetect']) & (~all_df['optselect'])]
df['sig_noise_ratio'] = df['data_name'].str.split('-').str[1].astype('int')
df['sig_noise_ratio'] = df['sig_noise_ratio'] / 10
# df.groupby(['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins', 'sig_noise_ratio']).mean()[['overall_accuracy']]
# df[df['seed'] == 5].groupby(['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins', 'sig_noise_ratio']).mean()[['overall_accuracy']]
# df.groupby(['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins']).mean()[['fw-f1mean']]
df[['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins', 'drift_width', 'overall_accuracy', 'seed']]
# %%
def fs_replace(fs):
    replace_dict = {
        'fisher': 'Fisher',
        'fisher_overall': 'Fisher Adjusted',
        'CacheMIHy': 'Gaussian MI',
        'Cachehistogram_MI': 'Histogram MI',
        'sketch_MI': 'Sketch MI',
        'sketch_covMI': 'Sketch MI + Redundancy',
    }
    return replace_dict[fs]
#%%

fig, ax = plt.subplots()
fp = 'cachehistogram'
fs = 'Cachehistogram_MI'
sm = 'metainfo'
def plot_feature_weights(fig, ax, fp, fs, sm, legend=True, stdev=False):
    # for b in sorted(df['fingerprint_bins'].unique(), reverse=True):
    # for b in [2, 3, 4, 5, 6, 7, 8, 9, 10, 100]:
    fp_cond = df['fingerprint_method'] == fp
    fs_cond = df['fs_method'] == fs
    sm_cond = df['similarity_option'] == sm
    bin_cond = df['fingerprint_bins'] == 100
    select_df = df[fp_cond & fs_cond & sm_cond & bin_cond]
    sns.lineplot(data=select_df, x='drift_width', y='overall_accuracy', ax=ax, label=fs_replace(fs), err_style=None if not stdev else 'bars')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Gradual Drift Width")
    ax.set_title(f"Performance across gradual drift")
    if legend:
        leg = ax.legend(title="Feature\nSelection", bbox_to_anchor=[1.0, 1.0], loc="upper left", frameon=False)
        leg._legend_box.align = "left"
    else:
        ax.get_legend().remove()

plot_feature_weights(fig, ax, fp, fs, sm)
plt.show()

# %%
fig, ax = plt.subplots()
for fp, fs, sm in [
                # ('cache', 'fisher', 'metainfo'), 
                ('cache', 'fisher_overall', 'metainfo'), 
                ('cache', 'CacheMIHy', 'metainfo'),
                ('cachehistogram', 'Cachehistogram_MI', 'metainfo'),
                ('cachesketch', 'sketch_MI', 'metainfo'),
                # ('cachesketch', 'sketch_MI', 'sketch'),
                ('cachesketch', 'sketch_covMI', 'metainfo'),
                # ('cachesketch', 'sketch_covMI', 'sketch'),
                ]:
    plot_feature_weights(fig, ax, fp, fs, sm)
plt.tight_layout()
# plt.show()
plt.savefig(f"gradual_drift_.pdf")

# %%
df['method'] = df['fingerprint_method'] + df['fs_method'] + df['similarity_option'] + df['fingerprint_bins'].astype(str)
df['run'] = df['seed'].astype(str) + df['sig_noise_ratio'].astype(str)
# %%
def replace_names(n):
    replaces = {
        "cachehistogramCachehistogram_MImetainfo100" : "Histogram MI k=100",
        "cachehistogramCachehistogram_MImetainfo2" : "Histogram MI k=2",
        "cacheCacheMIHymetainfo100" : "Gaussian MI k=100",
        "cacheCacheMIHymetainfo2" : "Gaussian MI k=2",
        "cachesketchsketch_MImetainfo100" : "Sketch MI k=100",
        "cachesketchsketch_MImetainfo2" : "Sketch MI k=2",
        "cachesketchsketch_covMImetainfo100" : "Sketch Redundancy MI k=100",
        "cachesketchsketch_covMImetainfo2" : "Sketch Redundancy MI k=2",
        "cachefishermetainfo2" : "Fisher Score",
        "cachefisher_overallmetainfo2" : "Fisher Adjusted",
    }
    return replaces[n]

for test_set in [
    ['cachehistogramCachehistogram_MImetainfo100', 'cachehistogramCachehistogram_MImetainfo2', 'cachefishermetainfo2'],
    ['cacheCacheMIHymetainfo100', 'cacheCacheMIHymetainfo2', 'cachefishermetainfo2'],
    ['cachesketchsketch_MImetainfo100', 'cachesketchsketch_MImetainfo2', 'cachefishermetainfo2'],
    ['cachesketchsketch_covMImetainfo100', 'cachesketchsketch_covMImetainfo2', 'cachefishermetainfo2']
]:
# select_df = df[df['method'].isin(['cachehistogramCachehistogram_MImetainfo100', 'cachehistogramCachehistogram_MImetainfo2', 'cachefishermetainfo2', 'cachefisher_overallmetainfo2'])]
# select_df = df[df['method'].isin(['cachehistogramCachehistogram_MImetainfo100', 'cachehistogramCachehistogram_MImetainfo2', 'cachefishermetainfo2'])]
# select_df = df[df['method'].isin(['cacheCacheMIHymetainfo100', 'cacheCacheMIHymetainfo2', 'cachefishermetainfo2', 'cachefisher_overallmetainfo2'])]
# select_df = df[df['method'].isin(['cachesketchsketch_MImetainfo100', 'cachesketchsketch_MImetainfo2', 'cachefishermetainfo2', 'cachefisher_overallmetainfo2'])]
# select_df = df[df['method'].isin(['cachesketchsketch_covMImetainfo100', 'cachesketchsketch_covMImetainfo2', 'cachefishermetainfo2', 'cachefisher_overallmetainfo2'])]
    print(test_set)
    select_df = df[df['method'].isin(test_set)]
    select_df = select_df[select_df['sig_noise_ratio'] <= 0.4]
    friedman_value_df = select_df.groupby(['method', 'run']).mean()['kappa'].unstack('method')
    friedman_values = friedman_value_df.to_numpy()
    friedman_ranked_df = friedman_value_df.rank(axis=1, ascending=False)
    friedman_ranked = friedman_ranked_df.to_numpy()
    names = [replace_names(n) for n in friedman_ranked_df.columns]
    avranks = friedman_ranked.mean(axis=0)
    avranks
    from scipy.stats import friedmanchisquare
    val, p = friedmanchisquare(*[friedman_values[:, c] for c in range(friedman_values.shape[1])])
    print(val, p)
    import Orange
    num_seeds = 10
    num_datasets = 8
    num_runs = num_seeds * num_datasets
    cd = Orange.evaluation.compute_CD(avranks, 80, alpha='0.1')
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
    plt.savefig(f"gradual_drift_{test_set}_sig.pdf")

#%%
friedman_value_df
# %%
