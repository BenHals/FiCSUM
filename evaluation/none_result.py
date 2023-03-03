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
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-none")
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
print(all_df.iloc[0]['isources'])
# %%
# df = all_df
df = all_df[all_df['isources'].isnull()]
# df = all_df[(all_df['data_name'] == 'SigNoiseGenerator-4') & (all_df['fs_method'] == 'CacheMIHy')]
# df = all_df[(~all_df['optdetect']) & (~all_df['optselect'])]
df['sig_noise_ratio'] = df['data_name'].str.split('-').str[1].astype('int')
df['sig_noise_ratio'] = df['sig_noise_ratio'] / 10
# df.groupby(['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins', 'sig_noise_ratio']).mean()[['overall_accuracy']]
# df[df['seed'] == 5].groupby(['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins', 'sig_noise_ratio']).mean()[['overall_accuracy']]
# df.groupby(['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins']).mean()[['fw-f1mean']]
df[['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins', 'sig_noise_ratio', 'overall_accuracy', 'seed', 'isources']]
# %%
def fs_replace(fs):
    replace_dict = {
        'fisher': 'Fisher',
        'fisher_overall': 'Fisher Adjusted',
        'CacheMIHy': 'Gaussian MI',
        'Cachehistogram_MI': 'Histogram MI',
        'sketch_MI': 'Sketch MI',
        'sketch_covMI': 'Sketch MI + Redundancy',
        'None': "No Feature Selection"
    }
    return replace_dict[fs]
#%%

fig, ax = plt.subplots()

def plot_feature_weights(fig, ax, fp, fs, sm, legend=True, stdev=False):
    for b in sorted(df['fingerprint_bins'].unique(), reverse=True):
    # for b in [2, 3, 4, 5, 6, 7, 8, 9, 10, 100]:
        fp_cond = df['fingerprint_method'] == fp
        fs_cond = df['fs_method'] == fs
        sm_cond = df['similarity_option'] == sm
        bin_cond = df['fingerprint_bins'] == b
        select_df = df[fp_cond & fs_cond & sm_cond & bin_cond]
        sns.lineplot(data=select_df, x='sig_noise_ratio', y='overall_accuracy', ax=ax, label=fs_replace(fs), err_style=None if not stdev else 'bars')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Signal-to-Noise Ratio")
    ax.set_title(f"Feature selection vs no feature selection")
    if legend:
        leg = ax.legend(title="Feature Selection", bbox_to_anchor=[1.0, 1.0], loc="upper left", frameon=False)
        leg._legend_box.align = "left"
    else:
        ax.get_legend().remove()

fp = 'cache'
fs = 'None'
sm = 'metainfo'
plot_feature_weights(fig, ax, fp, fs, sm)
fp = 'cache'
fs = 'fisher'
sm = 'metainfo'
plot_feature_weights(fig, ax, fp, fs, sm)
fp = 'cache'
fs = 'fisher_overall'
sm = 'metainfo'
plot_feature_weights(fig, ax, fp, fs, sm)
# plt.show()
plt.savefig("No_feature_selection_results.pdf", bbox_inches='tight')

# %%
#%%
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
sns.set()
sns.set_style('whitegrid')
sns.set_context('paper')
# %%
replace_dict = {
    'fisher': 'Fisher',
    'fisher_overall': 'Fisher Adjusted',
    'CacheMIHy': 'Gaussian MI',
    'Cachehistogram_MI': 'Histogram MI',
    'sketch_MI': 'Sketch MI',
    'sketch_covMI': 'Sketch MI + Redundancy',
    'None': "No Feature Selection"
}
def fs_replace(fs):
    return replace_dict[fs]
#%%
real_cmc_data_dir = pathlib.Path(r"S:\PhD\results\cmc")
real_UCI_Wine_data_dir = pathlib.Path(r"S:\PhD\results\UCI-Wine")
real_AQSex_data_dir = pathlib.Path(r"S:\PhD\results\AQSex")

def get_data(data_path, name):
    results_files = data_path.rglob("result*")
    results = []
    for rf in results_files:
        result = json.load(rf.open('r'))
        if 'feature_weights' in result:
            for k in result['feature_weights']:
                result[f"fw-{k}"] = result['feature_weights'][k]
        result['exp_name'] = name
        results.append(result)
    df = pd.DataFrame(results)
    opt_df = df[df['optselect'] & df['optdetect']]
    opt_df['exp_name'] = f"{name} Opt"
    nonopt_df = df[~df['optselect'] & ~df['optdetect']]
    nonopt_df['exp_name'] = f"{name} nonOpt"

    return df, opt_df, nonopt_df

cmc_all_df, cmc_opt_df, cmc_non_opt_df = get_data(real_cmc_data_dir, "CMC")
UCI_Wine_all_df, UCI_Wine_opt_df, UCI_Wine_non_opt_df = get_data(real_UCI_Wine_data_dir, "UCI-Wine")
AQSex_all_df, AQSex_opt_df, AQSex_non_opt_df = get_data(real_AQSex_data_dir, "AQSex")
#%%
cmc_all_df['fs_method'].unique()
# %%
def mean_stdev(x):
    return f"{np.mean(x):.2f} ({np.std(x):.2f})"
df = pd.concat((cmc_non_opt_df, UCI_Wine_non_opt_df,AQSex_non_opt_df))
# df = fweight_df
select_sim_opts = df['similarity_option'] == 'metainfo'
select_size_opts = df['fingerprint_bins'].isin((2, 25, 50, 75, 100))
df = df[select_sim_opts & select_size_opts]
df['fs_method'] = df['fs_method'].replace(replace_dict)
table_df = df.groupby(['exp_name', 'fs_method', 'fingerprint_bins']).mean()[['kappa', 'driftdetect_250_kappa', 'GT_mean_f1']].unstack('fingerprint_bins')
table_df = df.groupby(['exp_name', 'fs_method', 'fingerprint_bins']).aggregate(mean_stdev)[['kappa',  'GT_mean_f1']].unstack('fingerprint_bins')
table_df

# %%
table_df.to_latex('real_data_table.txt', float_format="{:0.2f}".format)
# %%
# for df in (cmc_non_opt_df, UCI_Wine_non_opt_df,AQSex_non_opt_df):
cmc_non_opt_df.columns
#%%
def replace_names(n):
    return n
metric = 'kappa'
# metric = 'GT_mean_f1'
name = "CMC"
# name = "UCI-Wine"
# name = "AQSex"
df = cmc_non_opt_df
# df = UCI_Wine_non_opt_df
# df = AQSex_non_opt_df

df['method'] = df['fs_method'] + df['fingerprint_bins'].astype(str)
friedman_values_df = df.groupby(['method', 'seed']).mean()[metric].unstack('method')
friedman_values = friedman_values_df.to_numpy()
friedman_ranked_df = friedman_values_df.rank(axis=1, ascending=False)
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
cd = Orange.evaluation.compute_CD(avranks, 15, alpha='0.1')
Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
plt.savefig(f"real_{metric}_{name}_sig.pdf")
# plt.show()
# %%
