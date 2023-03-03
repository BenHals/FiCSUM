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
}
def fs_replace(fs):
    return replace_dict[fs]
#%%
real_cmc_data_dir = pathlib.Path(r"S:\PhD\results\cmc")
real_UCI_Wine_data_dir = pathlib.Path(r"S:\PhD\results\UCI-Wine")
real_AQSex_data_dir = pathlib.Path(r"S:\PhD\results\AQSex")
real_Arabic_data_dir = pathlib.Path(r"S:\PhD\results\Arabic")

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
Arabic_all_df, Arabic_opt_df, Arabic_non_opt_df = get_data(real_Arabic_data_dir, "Arabic")
#%%
cmc_all_df.columns
# %%
def mean_stdev(x):
    return f"{np.mean(x):.2f} ({np.std(x):.2f})"
df = pd.concat((cmc_non_opt_df, UCI_Wine_non_opt_df,AQSex_non_opt_df, Arabic_non_opt_df))
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
