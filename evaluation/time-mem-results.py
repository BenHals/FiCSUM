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
signoise_data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\sig-noise-6")
fweight_data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\feature-weight-test-opt")
fweight_nonopt_data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\feature-weight-test-nonopt")
fweight_nonopt_data_dir2 = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\feature-weight-test-nonopt-2")
real_cmc_data_dir = pathlib.Path(r"S:\PhD\results\cmc")
real_UCI_Wine_data_dir = pathlib.Path(r"S:\PhD\results\UCI-Wine")
AQSex_data_dir = pathlib.Path(r"S:\PhD\results\AQSex")
# Arabic_data_dir = pathlib.Path(r"S:\PhD\results\Arabic")

results_files = signoise_data_dir.rglob("result*")
results = []
for rf in results_files:
    result = json.load(rf.open('r'))
    if 'feature_weights' in result:
        for k in result['feature_weights']:
            result[f"fw-{k}"] = result['feature_weights'][k]
    result['exp_name'] = "Signal to Noise"
    results.append(result)
signoise_df = pd.DataFrame(results)
results_files = fweight_data_dir.rglob("result*")
results = []
for rf in results_files:
    result = json.load(rf.open('r'))
    if 'feature_weights' in result:
        for k in result['feature_weights']:
            result[f"fw-{k}"] = result['feature_weights'][k]
    result['exp_name'] = "Feature Weight opt"
    results.append(result)
fweight_opt_df = pd.DataFrame(results)
results_files = fweight_nonopt_data_dir.rglob("result*")
results = []
for rf in results_files:
    result = json.load(rf.open('r'))
    if 'feature_weights' in result:
        for k in result['feature_weights']:
            result[f"fw-{k}"] = result['feature_weights'][k]
    result['exp_name'] = "Feature Weight nonopt"
    results.append(result)
results_files = fweight_nonopt_data_dir2.rglob("result*")
for rf in results_files:
    result = json.load(rf.open('r'))
    if 'feature_weights' in result:
        for k in result['feature_weights']:
            result[f"fw-{k}"] = result['feature_weights'][k]
    result['exp_name'] = "Feature Weight nonopt"
    results.append(result)
fweight_nonopt_df = pd.DataFrame(results)

results_files = real_cmc_data_dir.rglob("result*")
results = []
for rf in results_files:
    result = json.load(rf.open('r'))
    if 'feature_weights' in result:
        for k in result['feature_weights']:
            result[f"fw-{k}"] = result['feature_weights'][k]
    result['exp_name'] = "CMC"
    results.append(result)
cmc_df = pd.DataFrame(results)
cmc_df_opt = cmc_df[cmc_df['optselect'] & cmc_df['optdetect']]
cmc_df_opt['exp_name'] = "CMC Opt"
cmc_df_nonopt = cmc_df[~cmc_df['optselect'] & ~cmc_df['optdetect']]
cmc_df_nonopt['exp_name'] = "CMC nonOpt"
cmc_df_opt.columns
results_files = real_UCI_Wine_data_dir.rglob("result*")
results = []
for rf in results_files:
    result = json.load(rf.open('r'))
    if 'feature_weights' in result:
        for k in result['feature_weights']:
            result[f"fw-{k}"] = result['feature_weights'][k]
    result['exp_name'] = "UCI-Wine"
    results.append(result)
UCI_Wine_df = pd.DataFrame(results)
# UCI_Wine_opt = UCI_Wine_df[UCI_Wine_df['optselect'] & UCI_Wine_df['optdetect']]
UCI_Wine_opt = UCI_Wine_df[UCI_Wine_df['optselect']]
UCI_Wine_opt['exp_name'] = "UCI-Wine Opt"
print((UCI_Wine_df['optdetect']).unique())
UCI_Wine_nonopt = UCI_Wine_df[~UCI_Wine_df['optselect'] & ~UCI_Wine_df['optdetect']]
UCI_Wine_nonopt['exp_name'] = "UCI-Wine nonOpt"
UCI_Wine_df.columns
results_files = AQSex_data_dir.rglob("result*")
results = []
for rf in results_files:
    result = json.load(rf.open('r'))
    if 'feature_weights' in result:
        for k in result['feature_weights']:
            result[f"fw-{k}"] = result['feature_weights'][k]
    result['exp_name'] = "AQSex"
    results.append(result)
AQSex_df = pd.DataFrame(results)
AQSex_opt = AQSex_df[AQSex_df['optselect']]
AQSex_opt['exp_name'] = "AQSex Opt"
AQSex_nonopt = AQSex_df[~AQSex_df['optselect'] & ~AQSex_df['optdetect']]
AQSex_nonopt['exp_name'] = "AQSex nonOpt"
# results_files = Arabic_data_dir.rglob("result*")
# results = []
# for rf in results_files:
#     result = json.load(rf.open('r'))
#     if 'feature_weights' in result:
#         for k in result['feature_weights']:
#             result[f"fw-{k}"] = result['feature_weights'][k]
#     result['exp_name'] = "Arabic"
#     results.append(result)
# Arabic_df = pd.DataFrame(results)
# Arabic_opt = Arabic_df[Arabic_df['optselect']]
# Arabic_opt['exp_name'] = "Arabic Opt"
# Arabic_nonopt = Arabic_df[~Arabic_df['optselect'] & ~Arabic_df['optdetect']]
# Arabic_nonopt['exp_name'] = "Arabic nonOpt"
#%%
cmc_df = cmc_df[cmc_df['similarity_option'] == 'metainfo']
cmc_df = cmc_df[cmc_df['fingerprint_bins'].isin((2, 25, 50, 75, 100))]
cmc_df['optselect']
# %%
def mean_stdev(x):
    x = x.dropna()
    return f"{np.mean(x):.2f} ({np.std(x):.2f})"
def mean_stdev_n(x):
    x = x.dropna()
    return f"{np.mean(x):.1f} ({np.std(x):.1f})"
#%%
df = pd.concat((signoise_df, fweight_opt_df,fweight_nonopt_df, cmc_df_opt, cmc_df_nonopt, UCI_Wine_opt, UCI_Wine_nonopt, AQSex_opt, AQSex_nonopt))
# df = fweight_df
select_sim_opts = df['similarity_option'] == 'metainfo'
select_size_opts = df['fingerprint_bins'].isin((25, 50, 75, 100))
df = df[select_sim_opts & select_size_opts]
df['fs_method'] = df['fs_method'].replace(replace_dict)
table_df = df.groupby(['exp_name', 'fs_method', 'fingerprint_bins']).aggregate(mean_stdev)[['peak_fingerprint_mem', 'overall_time']].unstack('fingerprint_bins')
table_df

# %%
table_df.to_latex('time_mem_table.txt', float_format="{:0.2f}".format)
#%%
# selector = df['fs_method'] == 'Sketch MI'
selector = df['fs_method'] == 'Histogram MI'
selector = selector & (df['exp_name'] == 'AQSex Opt')
selector = selector & (df['fingerprint_bins'] == 25)
test_df = df[selector]
test_df[['peak_fingerprint_mem', 'average_fingerprint_mem', 'seed', 'data_name']]
# %%
df = pd.concat((signoise_df, fweight_opt_df,fweight_nonopt_df, cmc_df_opt, cmc_df_nonopt, UCI_Wine_opt, UCI_Wine_nonopt, AQSex_opt, AQSex_nonopt))
# df = fweight_df
select_sim_opts = df['similarity_option'] == 'metainfo'
select_size_opts = df['fingerprint_bins'].isin((25, 100))
df = df[select_sim_opts & select_size_opts]
df['fs_method'] = df['fs_method'].replace(replace_dict)
table_df = df.groupby(['exp_name', 'fs_method', 'fingerprint_bins']).aggregate(mean_stdev_n)[['peak_fingerprint_mem', 'overall_time']].unstack('fingerprint_bins')
table_df
# %%
table_df.to_latex('time_mem_table_narrow3.txt', float_format="{:0.1f}".format)

# %%
