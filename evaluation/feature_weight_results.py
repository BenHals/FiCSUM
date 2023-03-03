#%%
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
sns.set()
sns.set_style('whitegrid')
# sns.set_context('paper')
sns.set_context('talk')
from matplotlib.ticker import FormatStrFormatter

#%%
# using_opt = False
using_opt=True
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\feature-weight-test-opt")
if not using_opt:
    data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\feature-weight-test-nonopt")
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
# %%
df = all_df
df.groupby(['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins']).mean()[['fw-f1mean', 'fw-f2mean', 'fw-f3mean', 'fw-f4mean', 'fw-f5mean', 'fw-f6mean', 'fw-f7mean', 'fw-f8mean', 'fw-f9mean']]
# df.groupby(['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins']).mean()[['fw-f1mean']]

# %%
df = all_df[['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins', 'fw-f1mean', 'fw-f2mean', 'fw-f3mean', 'fw-f4mean', 'fw-f5mean', 'fw-f6mean', 'fw-f7mean', 'fw-f8mean', 'fw-f9mean']]
df = pd.melt(df, id_vars=['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins'])
df
# %%
def fs_replace(fs):
    replace_dict = {
        'fisher': 'Fisher',
        'fisher_overall': 'Fisher Adjusted',
        'CacheMIHy': 'Gaussian MI',
        'Cachehistogram_MI': 'Histogram MI',
        'sketch_MI': 'Sketch MI',
        'sketch_covMI': 'Sketch MI + Redundancy',
        'sketch_covMI_weighted': 'W Sketch MI + Redundancy',
    }
    return replace_dict[fs]
#%%

fig, ax = plt.subplots()
fp = 'cachehistogram'
fs = 'Cachehistogram_MI'
# fp = 'cachesketch'
# fs = 'sketch_covMI_weighted'
sm = 'metainfo'
def plot_feature_weights(fig, ax, fp, fs, sm, legend=True, stdev=False):
    # for b in sorted(df['fingerprint_bins'].unique(), reverse=True):
    for b in [2, 3, 4, 5, 6, 7, 8, 9, 10, 100]:
        fp_cond = df['fingerprint_method'] == fp
        fs_cond = df['fs_method'] == fs
        sm_cond = df['similarity_option'] == sm
        bin_cond = df['fingerprint_bins'] == b
        select_df = df[fp_cond & fs_cond & sm_cond & bin_cond]
        select_df['variable'] = select_df['variable'].str.split('-').str[1].str[:-4]
        sns.lineplot(data=select_df, x='variable', y='value', ax=ax, label=b, err_style=None if not stdev else 'band')
    ax.set_ylabel("Feature Weight")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{fs_replace(fs)}")
    if legend:
        leg = ax.legend(title="Approximation\nSize", bbox_to_anchor=[1.0, 1.0], loc="upper left", frameon=False)
        leg._legend_box.align = "left"
    else:
        ax.get_legend().remove()

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
    plot_feature_weights(fig, ax, fp, fs, sm, stdev=(not using_opt))
    plt.tight_layout()
    plt.savefig(f"feature_weight_plot_l-{fp}-{fs}-{fp}-opt-{using_opt}_v2.pdf")
# %%
fig, ax = plt.subplots(3, 2, sharex=True, figsize=(5, 5))
for i, (fp, fs, sm) in enumerate([('cache', 'fisher', 'metainfo'), 
                ('cache', 'fisher_overall', 'metainfo'), 
                ('cache', 'CacheMIHy', 'metainfo'),
                ('cachehistogram', 'Cachehistogram_MI', 'metainfo'),
                ('cachesketch', 'sketch_MI', 'metainfo'),
                # ('cachesketch', 'sketch_MI', 'sketch'),
                ('cachesketch', 'sketch_covMI', 'metainfo'),
                # ('cachesketch', 'sketch_covMI', 'sketch'),
                ]):
    r = i // 2
    c = i % 2
    plot_feature_weights(fig, ax[r, c], fp, fs, sm, legend=False, stdev=(not using_opt))
handles, labels = ax[2, 1].get_legend_handles_labels()
fig.legend(handles, labels, title="Approximation\nSize", bbox_to_anchor=[1.0, 1.0], loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(f"feature_weight_plot_l-set-opt-{using_opt}_v2.pdf")
# %%
fig, ax = plt.subplots(2, 3, sharex=True, figsize=(7.5, 4))
for i, (fp, fs, sm) in enumerate([('cache', 'fisher', 'metainfo'), 
                ('cache', 'fisher_overall', 'metainfo'), 
                ('cache', 'CacheMIHy', 'metainfo'),
                ('cachehistogram', 'Cachehistogram_MI', 'metainfo'),
                ('cachesketch', 'sketch_MI', 'metainfo'),
                # ('cachesketch', 'sketch_MI', 'sketch'),
                ('cachesketch', 'sketch_covMI', 'metainfo'),
                # ('cachesketch', 'sketch_covMI', 'sketch'),
                ]):
    r = i % 2
    c = i // 2
    # plot_feature_weights(fig, ax[r, c], fp, fs, sm, legend=False, stdev=(not using_opt))
    plot_feature_weights(fig, ax[r, c], fp, fs, sm, legend=False, stdev=True)
handles, labels = ax[1, 1].get_legend_handles_labels()
fig.legend(handles, labels, title="Approximation\nSize", bbox_to_anchor=[1.0, 1.0], loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(f"feature_weight_plot_l-set_horizontal_2-opt-{using_opt}-stdev_v2.pdf", bbox_inches='tight')
# %%
fig, ax = plt.subplots(1, 6, sharex=True, figsize=(20, 3.5))
for i, (fp, fs, sm) in enumerate([('cache', 'fisher', 'metainfo'), 
                ('cache', 'fisher_overall', 'metainfo'), 
                ('cache', 'CacheMIHy', 'metainfo'),
                ('cachehistogram', 'Cachehistogram_MI', 'metainfo'),
                ('cachesketch', 'sketch_MI', 'metainfo'),
                # ('cachesketch', 'sketch_MI', 'sketch'),
                ('cachesketch', 'sketch_covMI', 'metainfo'),
                # ('cachesketch', 'sketch_covMI', 'sketch'),
                ]):
    # r = i % 2
    # c = i // 2
    # plot_feature_weights(fig, ax[r, c], fp, fs, sm, legend=False, stdev=(not using_opt))
    # plot_feature_weights(fig, ax[r, c], fp, fs, sm, legend=False, stdev=True)
    ax[i].tick_params(axis='y', which='major', pad=-2)
    plot_feature_weights(fig, ax[i], fp, fs, sm, legend=False, stdev=True)
    if i != 0:
        ax[i].set_ylabel("")
        # ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# handles, labels = ax[1, 1].get_legend_handles_labels()
handles, labels = ax[5].get_legend_handles_labels()
labels = ["Approximation Size: ", *labels]
ph = [plt.plot([],marker="", ls="")[0]] # Canvas
handles = ph + handles
# fig.legend(handles, labels, title="Approximation\nSize", bbox_to_anchor=[1.0, 1.0], loc="upper left", frameon=False)
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=[0.5, 0.1], ncol=12, frameon=False)
plt.tight_layout()
fig.subplots_adjust(wspace=0.25)
plt.savefig(f"feature_weight_plot_l-set_horizontal_2-opt-{using_opt}-stdev_v4.pdf", bbox_inches='tight')

# %%
fp = 'cachehistogram'
fs = 'Cachehistogram_MI'
# fp = 'cachesketch'
# fs = 'sketch_covMI_weighted'
sm = 'metainfo'
    # # for b in sorted(df['fingerprint_bins'].unique(), reverse=True):
    # for b in [2, 3, 4, 5, 6, 7, 8, 9, 10, 100]:
for i, (fp, fs, sm) in enumerate([('cache', 'fisher', 'metainfo'), 
                ('cache', 'fisher_overall', 'metainfo'), 
                ('cache', 'CacheMIHy', 'metainfo'),
                ('cachehistogram', 'Cachehistogram_MI', 'metainfo'),
                ('cachesketch', 'sketch_MI', 'metainfo'),
                # ('cachesketch', 'sketch_MI', 'sketch'),
                ('cachesketch', 'sketch_covMI', 'metainfo'),
                # ('cachesketch', 'sketch_covMI', 'sketch'),
                ]):
    fp_cond = df['fingerprint_method'] == fp
    fs_cond = df['fs_method'] == fs
    sm_cond = df['similarity_option'] == sm
    bin_cond = df['fingerprint_bins'].isin([2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
    select_df = df[fp_cond & fs_cond & sm_cond & bin_cond]
    select_df['variable'] = select_df['variable'].str.split('-').str[1].str[:-4]
    def mean_std(x):
        return f"{np.mean(x):.2f} ({np.std(x):.2f})"
    select_df = select_df.groupby(['variable', 'fingerprint_bins']).aggregate(mean_std).unstack('variable')
    select_df.to_latex(f"feature_weight_results-{fp}-{fs}-{fp}-opt-{using_opt}.txt")
# %%
# select_df.to_latex(f"feature_weight_results")
# %%
