#%%
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
sns.set()

#%%
# data_dir = pathlib.Path(r"S:\PhD\results\FingerprintExtension\testH")
data_dir = pathlib.Path(r"S:\PhD\results\FingerprintExtension\test")
# data_dir = pathlib.Path(r"S:\PhD\results\FingerprintExtension\testOptDetect")
data_dir = pathlib.Path(r"S:\PhD\results\FingerprintExtension\testOptSelect")
data_dir = pathlib.Path(r"S:\PhD\results\FingerprintExtension\testH2")
data_dir = pathlib.Path(r"S:\PhD\results\FingerprintExtension\testUU3")
data_dir = pathlib.Path(r"S:\PhD\results\FingerprintExtension\testNU")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\testNB")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\testHP23-v2")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\test-v2\ConceptFingerprint-master-bfa3502\RTREESAMPLE-NB\6400")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\test-v2\ConceptFingerprint-master-bfa3502\RTREESAMPLE-NB\8777")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\test-v5")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\test-v7")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\test-v12-s")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\test-v15-s-alternating")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\test-v20-s-alternating\ConceptFingerprint-master-370ea36\RTREESAMPLE-NB\4222")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\test-binperf-2\ConceptFingerprint-master-1efd58e\RTREESAMPLE-NB\42")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\test-binperf-3")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\test-binperf-3\ConceptFingerprint-master-86e394c\RTREESAMPLE-NB\34")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\test-binperf-4")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\feature-weight-test-nonopt")
data_dir = pathlib.Path(r"S:\PhD\Packages\ConceptFingerprint\output\feature-weight-test-opt")
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
all_df['feature_weights']
# %%
# df = all_df[all_df['data_name'] == "RTREE"]
# df = all_df[all_df['data_name'] == "RTREESAMPLEHP-23"]
df = all_df
df = df[df['concept_max'] == 10]
# df.groupby(['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins']).mean()
df.groupby(['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins']).mean()[['overall_accuracy', 'overall_time', 'overall_mem', 'peak_fingerprint_mem', 'average_fingerprint_mem', 'GT_mean_f1']]
# %%
df.groupby(['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins']).mean()[['fw-f1mean', 'fw-f2mean', 'fw-f3mean', 'fw-f4mean', 'fw-f5mean', 'fw-f6mean', 'fw-f7mean', 'fw-f8mean', 'fw-f9mean']]
# df.groupby(['fingerprint_method', 'fs_method', 'similarity_option', 'fingerprint_bins']).mean()[['fw-f1mean']]

# %%
