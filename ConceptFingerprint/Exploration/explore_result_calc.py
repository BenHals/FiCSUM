#%%
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt


log_path = pathlib.Path('S:\\output\\expDefault\\ConceptFingerprint-optimalClassifierClean-6387b8a\\Arabic\\7076\\run_-2810774206940565985_0.csv')
log_path = pathlib.Path('S:\\output\\expDefault\\ConceptFingerprint-optimalClassifierClean-6dbb604\\Arabic\\7076\\run_-1167756462641436164_0.csv')
# log_path = pathlib.Path('S:\\output\\expDefault\\ConceptFingerprint-optimalClassifierClean-6dbb604\\Arabic\\7076\\run_2073713035361685860_0.csv')
# log_path = pathlib.Path('S:\\output\\expDefault\\ConceptFingerprint-optimalClassifierClean-499113f\\Arabic\\7076\\run_7071243073281919050_0.csv')
# log_path = pathlib.Path('S:\\output\\expDefault\\ConceptFingerprint-optimalClassifierClean-27a1e82\\Arabic\\7076\\run_6406213975419943930_0.csv')
# log_path = pathlib.Path('S:\\output\\expDefault\\ConceptFingerprint-optimalClassifierClean-380da76\\Arabic\\7076\\run_8505724373964717276_0.csv')
# log_path = pathlib.Path('S:\\output\\t\\ConceptFingerprint-optimalClassifierClean-490f324\\HPLANE\\1\\run_2695549778667210504_0.csv')
# log_path = pathlib.Path('S:\\output\\t\\ConceptFingerprint-optimalClassifierClean-3db3c47\\HPLANE\\1\\run_-4091275052440908650_0.csv')
log_path = pathlib.Path('S:\\output\\timing\\ConceptFingerprint-optimalClassifierClean-057c9ff\\AQTemp\\1\\run_-2939791174272254135_0.csv')
log_path = pathlib.Path('S:\\output\\timing\\ConceptFingerprint-HEAD-490f324\\AQTemp\\1\\run_2279087236836059086_0.csv')
log_path = pathlib.Path('S:\\output\\timing\\ConceptFingerprint-optimalClassifierClean-057c9ff\\HPLANE\\1\\run_6723233005565437269_0.csv')
log_path = pathlib.Path('S:\\output\\timing\\ConceptFingerprint-HEAD-490f324\\HPLANE\\1\\run_-3112749642879619424_0.csv')
log_path = pathlib.Path('S:\\output\\errorCheck\\ConceptFingerprint-optimalClassifierClean-8bc958f\\HPLANE\\1530\\run_246161004414994684_0.csv')
log_path = pathlib.Path('S:\\output\\errorCheck\\ConceptFingerprint-optimalClassifierClean-8bc958f\\HPLANE\\1530\\run_-7497134848541758984_0.csv')
log_path = pathlib.Path('S:\\output\\errorCheck\\ConceptFingerprint-optimalClassifierClean-3ecfcaf\\HPLANE\\1530\\run_7890599434695463265_0.csv')
log_path = pathlib.Path('S:\\output\\errorCheck\\ConceptFingerprint-optimalClassifierClean-3ecfcaf\\HPLANE\\1530\\run_-3457176982106046799_0.csv')
log_path = pathlib.Path('S:\\output\\errorCheck\\ConceptFingerprint-optimalClassifierClean-e195f66\\HPLANESAMPLE\\1530\\run_776106246452220848_0.csv')
# log_path = pathlib.Path('S:\\output\\errorCheck\\ConceptFingerprint-optimalClassifierClean-3ecfcaf\\HPLANESAMPLE\\1530\\run_-187911071823377204_0.csv')
log_path = pathlib.Path('S:\\output\\errorCheck\\ConceptFingerprint-optimalClassifierClean-e195f66\\AQSex\\56\\run_-3432371489079711821_0.csv')
log_path = pathlib.Path('S:\\output\\errorCheck\\ConceptFingerprint-optimalClassifierClean-5d33276\\RTREESAMPLE\\56\\run_-1038187341733410033_0.csv')

log = pd.read_csv(log_path)
# %%
log.head()
# %%
ground_truth = log['ground_truth_concept'].fillna(method='ffill').astype(int).values
system = log['active_model'].fillna(method='ffill').astype(int).values
gt_values, gt_total_counts = np.unique(ground_truth, return_counts = True)
sys_values, sys_total_counts = np.unique(system, return_counts = True)
matrix = np.array([ground_truth, system]).transpose()
recall_values = {}
precision_values = {}
gt_results = {}
sys_results = {}
overall_results = {
    'Max Recall': 0,
    'Max Precision': 0,
    'Precision for Max Recall': 0,
    'Recall for Max Precision': 0,
    'GT_mean_f1' : 0,
    'GT_mean_recall':0,
    'GT_mean_precision':0,
    'MR by System': 0,
    'MP by System': 0,
    'PMR by System': 0,
    'RMP by System': 0,
    'MODEL_mean_f1': 0,
    'MODEL_mean_recall': 0,
    'MODEL_mean_precision': 0,
    'Num Good System Concepts': 0,
}
gt_proportions = {}
sys_proportions = {}

for gt_i, gt in enumerate(gt_values):
    gt_total_count = gt_total_counts[gt_i]
    gt_mask = matrix[matrix[:,0] == gt]
    sys_by_gt_values, sys_by_gt_counts = np.unique(gt_mask[:, 1], return_counts = True)
    gt_proportions[gt] = gt_mask.shape[0] / matrix.shape[0]
    max_recall = None
    max_recall_sys = None
    max_precision = None
    max_precision_sys = None
    max_f1 = None
    max_f1_sys = None
    max_f1_recall = None
    max_f1_precision = None
    for sys_i,sys in enumerate(sys_by_gt_values):
        sys_by_gt_count = sys_by_gt_counts[sys_i]
        sys_total_count = sys_total_counts[sys_values.tolist().index(sys)]
        if gt_total_count != 0:
            recall = sys_by_gt_count / gt_total_count
        else:
            recall = 1

        recall_values[(gt, sys)] = recall

        sys_proportions[sys] = sys_total_count / matrix.shape[0]
        if sys_total_count != 0:
            precision = sys_by_gt_count / sys_total_count
        else:
            precision = 1
        precision_values[(gt, sys)] = precision

        f1 = 2 * ((recall * precision) / (recall + precision))

        if max_recall == None or recall > max_recall:
            max_recall = recall
            max_recall_sys = sys
        if max_precision == None or precision > max_precision:
            max_precision = precision
            max_precision_sys = sys
        if max_f1 == None or f1 > max_f1:
            max_f1 = f1
            max_f1_sys = sys
            max_f1_recall = recall
            max_f1_precision = precision
    precision_max_recall = precision_values[(gt, max_recall_sys)]
    recall_max_precision = recall_values[(gt, max_precision_sys)]
    gt_result = {
        'Max Recall': max_recall,
        'Max Precision': max_precision,
        'Precision for Max Recall': precision_max_recall,
        'Recall for Max Precision': recall_max_precision,
        'f1': max_f1,
        'recall': max_f1_recall,
        'precision': max_f1_precision,
    }
    gt_results[gt] = gt_result
    overall_results['Max Recall'] += max_recall
    overall_results['Max Precision'] += max_precision
    overall_results['Precision for Max Recall'] += precision_max_recall
    overall_results['Recall for Max Precision'] += recall_max_precision
    overall_results['GT_mean_f1'] += max_f1
    overall_results['GT_mean_recall'] += max_f1_recall
    overall_results['GT_mean_precision'] += max_f1_precision


for sys in sys_values:
    max_recall = None
    max_recall_gt = None
    max_precision = None
    max_precision_gt = None
    max_f1 = None
    max_f1_sys = None
    max_f1_recall = None
    max_f1_precision = None
    for gt in gt_values:
        if (gt, sys) not in recall_values:
            continue
        if (gt, sys) not in precision_values:
            continue
        recall = recall_values[(gt, sys)]
        precision = precision_values[(gt, sys)]

        f1 = 2 * ((recall * precision) / (recall + precision))

        if max_recall == None or recall > max_recall:
            max_recall = recall
            max_recall_gt = gt
        if max_precision == None or precision > max_precision:
            max_precision = precision
            max_precision_gt = gt
        if max_f1 == None or f1 > max_f1:
            max_f1 = f1
            max_f1_sys = sys
            max_f1_recall = recall
            max_f1_precision = precision

    precision_max_recall = precision_values[(max_recall_gt, sys)]
    recall_max_precision = recall_values[(max_precision_gt, sys)]   
    sys_result = {
        'Max Recall': max_recall,
        'Max Precision': max_precision,
        'Precision for Max Recall': precision_max_recall,
        'Recall for Max Precision': recall_max_precision,
        'f1': max_f1
    }
    sys_results[sys] = sys_result     
    overall_results['MR by System'] += max_recall * sys_proportions[sys]
    overall_results['MP by System'] += max_precision * sys_proportions[sys]
    overall_results['PMR by System'] += precision_max_recall * sys_proportions[sys]
    overall_results['RMP by System'] += recall_max_precision * sys_proportions[sys]
    overall_results['MODEL_mean_f1'] += max_f1 * sys_proportions[sys]
    overall_results['MODEL_mean_recall'] += max_f1_recall * sys_proportions[sys]
    overall_results['MODEL_mean_precision'] += max_f1_precision * sys_proportions[sys]
    if max_recall > 0.75 and precision_max_recall > 0.75:
        overall_results['Num Good System Concepts'] += 1

# Get average over concepts by dividing by number of concepts
# Don't need to average over models as we already multiplied by proportion.
overall_results['Max Recall'] /= gt_values.size
overall_results['Max Precision'] /= gt_values.size
overall_results['Precision for Max Recall'] /= gt_values.size
overall_results['Recall for Max Precision'] /= gt_values.size
overall_results['GT_mean_f1'] /= gt_values.size
overall_results['GT_mean_recall'] /= gt_values.size
overall_results['GT_mean_precision'] /= gt_values.size
overall_results['GT_to_MODEL_ratio'] = overall_results['Num Good System Concepts'] / len(gt_values)

for k,v in overall_results.items():
    print(f"{k}:{v}")
# %%
models = log['active_model'].unique()
# For each model, we get the timestamps where it was active and we have a reading.
# Since we store as a ; seperated list, we split this out.
# Model as just stored as their index, but this works as a unique ID
divergences = {}
active_model_mean_divergences = {}
mean_divergence = []
divencence_list = []
for active_model_id in models:
    divergences[active_model_id] = {}
    active_model_mean_divergences[active_model_id] = 0
    comparisons_made = 0
    all_state_active_similarity = log.loc[log['active_model'] == active_model_id]['all_state_active_similarity'].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
    
    # For each comparison model, we get the readings where it was active.
    standard_deviations = []
    for comparison_model_id in models:
        if comparison_model_id == active_model_id:
            continue
        comparison_df = all_state_active_similarity[[active_model_id, comparison_model_id]].dropna()
        standard_deviations.append(comparison_df[active_model_id].dropna().std())
        standard_deviations.append(comparison_df[comparison_model_id].dropna().std())
    scale = np.mean(standard_deviations)
    print(f"{active_model_id}: {standard_deviations}")
    for comparison_model_id in models:
        if comparison_model_id == active_model_id:
            continue
        comparison_df = all_state_active_similarity[[active_model_id, comparison_model_id]].dropna()
        if comparison_df.shape[0] < 1:
            continue
        comparison_df['divergence'] = (comparison_df[comparison_model_id] - comparison_df[active_model_id])
        avg_divergence = comparison_df['divergence'].sum() / comparison_df.shape[0]
        # scale = all_state_active_similarity[active_model_id].std()
        # print(f"{all_state_active_similarity[active_model_id]}")
        print(f"{active_model_id}: {scale}")

        scaled_avg_divergence = avg_divergence / scale
        divencence_list.append(scaled_avg_divergence)
        divergences[active_model_id][comparison_model_id] = scaled_avg_divergence
        active_model_mean_divergences[active_model_id] += scaled_avg_divergence
        comparisons_made += 1
    active_model_mean_divergences[active_model_id] = active_model_mean_divergences[active_model_id] / comparisons_made
    mean_divergence.append(active_model_mean_divergences[active_model_id])

for active_model_id in models:
    for comparison_model_id in models:
        if active_model_id in divergences and comparison_model_id in divergences[active_model_id]:
            divergences[active_model_id][comparison_model_id] = (divergences[active_model_id][comparison_model_id] - min(divencence_list)) / (max(divencence_list) - min(divencence_list))
    active_model_mean_divergences[active_model_id] = (active_model_mean_divergences[active_model_id] - min(divencence_list)) / (max(divencence_list) - min(divencence_list))
mean_divergence = (np.mean(mean_divergence) - min(divencence_list)) / (max(divencence_list) - min(divencence_list))
print(divergences)
print(active_model_mean_divergences)
print(mean_divergence)
# %%
plt.figure(figsize=(20,20))
similarity_plots = {}
model_ids = log['active_model'].unique()
# print(model_ids)
# print(log['all_state_buffered_similarity'].replace('-1', np.nan).replace(-1, np.nan).dropna())
# active_similarity = log['all_state_buffered_similarity'].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
active_similarity = log['all_state_active_similarity'].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
# print(active_similarity)

active_concepts = log['ground_truth_concept'].fillna(method='ffill').values
active_models = log['active_model'].values

values = np.concatenate([active_similarity[m].dropna().values for m in active_similarity.columns])
max_similarity = np.percentile(values, 90)
min_similarity = min(values)
print(max_similarity)
print(min_similarity)

for m_id in model_ids:
    model_observations = active_similarity[m_id].dropna()
    x = list(model_observations.index.values)
    y = list(np.clip(model_observations.values, min_similarity, max_similarity))
    plt.plot(x, y)

current_concept = int(active_concepts[0])
concept_start = 0
index = 0
for c in active_concepts[1:]:
    if int(c) != current_concept:
        plt.hlines(y = -0.01, xmin = concept_start, xmax = index, colors = "C{}".format(current_concept))
        current_concept = int(c)
        concept_start = index
    index += 1
plt.hlines(y = -0.01, xmin = concept_start, xmax = index, colors = "C{}".format(current_concept))

current_model = int(active_models[0])
model_start = 0
index = 0
for c in active_models[1:]:
    if int(c) != current_model:
        plt.hlines(y = -0.005, xmin = model_start, xmax = index, colors = "C{}".format(current_model))
        current_model = int(c)
        model_start = index
    index += 1
plt.hlines(y = -0.005, xmin = model_start, xmax = index, colors = "C{}".format(current_model))

current_model = active_models[0]
index = 0
for m in active_models[1:]:
    if m != current_model:
        plt.vlines(x = index, ymin = -0.01, ymax = -0.1, colors = 'red')
        current_model = m
    index += 1

# plt.show()
# %%
models = log['active_model'].unique()
all_state_active_similarity = log['all_state_active_similarity'].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
max_similarity = None
min_similarity = None

values = np.concatenate([all_state_active_similarity[m].dropna().values for m in all_state_active_similarity.columns])
# for m in models:
#     if values is None:
#         values = all_state_active_similarity[m].values
#     else:
#         values = values + all_state_active_similarity[m].values


# max_val = np.percentile(all_state_active_similarity[m].values, 50)
# # max_val = 0
# min_val = all_state_active_similarity[m].min()
# max_similarity = max(max_similarity, max_val) if max_similarity is not None else max_val
# min_similarity = min(min_similarity, min_val) if min_similarity is not None else min_val
max_similarity = np.percentile(values, 90)
min_similarity = min(values)
print(max_similarity)
print(min_similarity)


model_changes = log['active_model'] != log['active_model'].shift(1).fillna(method='bfill')
chunk_masks = model_changes.cumsum()
chunks = chunk_masks.unique()
divergences = {}
active_model_mean_divergences = {}
mean_divergence = []
all_chunks = None
for chunk in chunks:
    chunk_mask = chunk_masks == chunk
    chunk_shift = chunk_mask.shift(50, fill_value=0)
    smaller_mask = chunk_mask & chunk_shift
    chunk_shift = chunk_mask.shift(-50, fill_value=0)
    smaller_mask = smaller_mask & chunk_shift
    all_state_active_similarity = log['all_state_active_similarity'].loc[smaller_mask].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
    if len(all_state_active_similarity.columns) < 2:
        continue
    if all_chunks is None:
        all_chunks = smaller_mask
    else:
        all_chunks = all_chunks | smaller_mask
print(all_chunks.sum())
for chunk in chunks:
    chunk_mask = chunk_masks == chunk
    chunk_shift = chunk_mask.shift(100, fill_value=0)
    smaller_mask = chunk_mask & chunk_shift
    chunk_shift = chunk_mask.shift(-100, fill_value=0)
    smaller_mask = smaller_mask & chunk_shift
    all_state_active_similarity = log['all_state_active_similarity'].loc[smaller_mask].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
    if all_state_active_similarity.shape[0] < 1:
        continue
    active_model = log['active_model'].loc[smaller_mask].unique()[0]
    for m in all_state_active_similarity.columns:
        # print(all_state_active_similarity[m])
        all_state_active_similarity[m] = (all_state_active_similarity[m] - min_similarity) / (max_similarity - min_similarity)
        all_state_active_similarity[m] = np.clip(all_state_active_similarity[m], 0, 1)
        # print(all_state_active_similarity[m])
    # stdevs = []
    # for m in all_state_active_similarity.columns:
    #     stdevs.append(all_state_active_similarity[m].std())
    # scale = np.mean(stdevs)
    chunk_proportion = smaller_mask.sum() / all_chunks.sum()
    # print(chunk_proportion)
    chunk_mean = []
    for m in all_state_active_similarity.columns:
        if m == active_model:
            continue
        if all_state_active_similarity[m].shape[0] < 2:
            continue
        # print("**")
        # print(all_state_active_similarity.shape[0])
        scale = np.mean([all_state_active_similarity[m].std(), all_state_active_similarity[active_model].std()])
        divergence = all_state_active_similarity[m] - all_state_active_similarity[active_model]
        avg_divergence = divergence.sum() / divergence.shape[0]
        # print(avg_divergence)
        scaled_avg_divergence = avg_divergence / scale
        # print(scaled_avg_divergence)
        scaled_avg_divergence *= chunk_proportion
        # print(scaled_avg_divergence)
        if active_model not in divergences:
            divergences[active_model] = {}
        if m not in divergences[active_model]:
            divergences[active_model][m] = scaled_avg_divergence
        if active_model not in active_model_mean_divergences:
            active_model_mean_divergences[active_model] = []
        active_model_mean_divergences[active_model].append(scaled_avg_divergence)
        chunk_mean.append(scaled_avg_divergence)
    if len(all_state_active_similarity.columns) > 1 and len(chunk_mean) > 0:
        mean_divergence.append(np.mean(chunk_mean))
print(mean_divergence)
mean_divergence = np.sum(mean_divergence)
for m in active_model_mean_divergences:
    active_model_mean_divergences[m] = np.sum(active_model_mean_divergences[m])
print(divergences)
print(active_model_mean_divergences)
print(mean_divergence)

# %%
