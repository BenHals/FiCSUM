#%%
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
sns.set()
sns.set_style('whitegrid')
# sns.set_style('white')
# sns.set_context('paper')
sns.set_context('talk')

#%%
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\expDefault\ConceptFingerprint-master-7603389")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-test")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-test-2")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-test-3")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-test-3")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-3")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-lock-shuffle")
data_dir = pathlib.Path(r"H:\PhD\PythonPackages\ConceptFingerprint\output\sig-noise-U-t-1")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-lock-3")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-lock-3\ConceptFingerprint-master-8f16e0e")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-lock-4")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-lock-5")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-3")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-3")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-lock-4\ConceptFingerprint-master-5ee43bb")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-6")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-graph")
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
df[['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins', 'sig_noise_ratio', 'overall_accuracy', 'seed']]
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
    for b in sorted(df['fingerprint_bins'].unique(), reverse=True):
    # for b in [2, 3, 4, 5, 6, 7, 8, 9, 10, 100]:
        fp_cond = df['fingerprint_method'] == fp
        fs_cond = df['fs_method'] == fs
        sm_cond = df['similarity_option'] == sm
        bin_cond = df['fingerprint_bins'] == b
        select_df = df[fp_cond & fs_cond & sm_cond & bin_cond]
        sns.lineplot(data=select_df, x='sig_noise_ratio', y='overall_accuracy', ax=ax, label=b, err_style=None if not stdev else 'bars')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Signal-to-Noise Ratio")
    ax.set_title(f"{fs_replace(fs)}")
    if legend:
        leg = ax.legend(title="Approximation\nSize", bbox_to_anchor=[1.0, 1.0], loc="upper left", frameon=False)
        leg._legend_box.align = "left"
    else:
        leg = ax.get_legend()
        if leg:
            leg.remove()

plot_feature_weights(fig, ax, fp, fs, sm)
plt.show()

# %%
for fp, fs, sm in [('cache', 'fisher', 'metainfo'), 
                ('cache', 'fisher_overall', 'metainfo'), 
                ('cache', 'CacheMIHy', 'metainfo'),
                ('cachehistogram', 'Cachehistogram_MI', 'metainfo'),
                ('cachesketch', 'sketch_MI', 'metainfo'),
                # ('cachesketch', 'sketch_MI', 'sketch'),
                ('cachesketch', 'sketch_covMI', 'metainfo'),
                # ('cachesketch', 'sketch_covMI', 'sketch'),
                ]:
    fig, ax = plt.subplots()
    plot_feature_weights(fig, ax, fp, fs, sm)
    plt.tight_layout()
    plt.savefig(f"sig_noise_plot_l-{fp}-{fs}-{fp}_v2.pdf")
# %%
# fig, ax = plt.subplots(2, 3, sharex=True, figsize=(10, 5), sharey=True)
fig, ax = plt.subplots(1, 6, sharex=True, figsize=(20, 3), sharey=True)
for i, (fp, fs, sm) in enumerate([('cache', 'fisher', 'metainfo'), 
                ('cache', 'fisher_overall', 'metainfo'), 
                ('cache', 'CacheMIHy', 'metainfo'),
                ('cachehistogram', 'Cachehistogram_MI', 'metainfo'),
                ('cachesketch', 'sketch_MI', 'metainfo'),
                # ('cachesketch', 'sketch_MI', 'sketch'),
                ('cachesketch', 'sketch_covMI', 'metainfo'),
                # ('cachesketch', 'sketch_covMI', 'sketch'),
                ]):
    print(i)
    r = i % 2
    c = i // 2
    print(r, c)
    # plot_feature_weights(fig, ax[r, c], fp, fs, sm, legend=False, stdev=True)
    plot_feature_weights(fig, ax[i], fp, fs, sm, legend=False, stdev=True)
# handles, labels = ax[1, 1].get_legend_handles_labels()
handles, labels = ax[5].get_legend_handles_labels()
# fig.legend(handles, labels, title="Approximation\nSize", bbox_to_anchor=[1.0, 1.0], loc="upper left", frameon=False)
# fig.legend(handles, labels, title="Approximation\nSize", loc="lower right", bbox_to_anchor=[0.9875, 0.1])
# fig.legend(handles, labels, title="Approximation\nSize", loc="lower right", bbox_to_anchor=[1.2, 0.6])
labels = ["Approximation Size: ", *labels]
ph = [plt.plot([],marker="", ls="")[0]] # Canvas
handles = ph + handles
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=[0.5, 0.125], ncol=3, frameon=False)
plt.tight_layout()
fig.subplots_adjust(wspace=0.05)
plt.savefig(f"sig_noise_plot_l-set_g-stdev_v4.pdf", bbox_inches="tight")

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
    plt.savefig(f"sig_noise_{test_set}_sig_v2.pdf")

#%%
friedman_value_df
# %%
