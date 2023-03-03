import pandas as pd
import numpy as np
import pathlib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import tqdm

sns.set()
my_parser = argparse.ArgumentParser()
my_parser.add_argument('--outputlocation', default="\divergence_plots", type=str)
my_parser.add_argument('--loglocation', default="\outputExperiments", type=str)
my_parser.add_argument('--experimentname', default="expDefault", type=str)
args = my_parser.parse_args()

base_path = pathlib.Path(args.loglocation)

output_path = pathlib.Path(args.outputlocation) / args.experimentname
output_path.mkdir(parents=True, exist_ok=True)

with (output_path / "run_args.txt").open('w+') as f:
    json.dump({'outputlocation': str(args.outputlocation), 'loglocation': str(args.loglocation), 'experimentname': str(args.experimentname)}, f)

log_files = list(base_path.rglob("run_*.csv"))

experiment_dfs = {}
for log_file in tqdm.tqdm(log_files):
    run_name = log_file.stem
    option_path = log_file.parent / f"{run_name}_options.txt"
    with option_path.open('r') as f:
        options = json.load(f)
    similarity_method = options['similarity_option']
    dataset = options['data_name']
    print(dataset)
    print(similarity_method)
    log = pd.read_csv(log_file)
    model_changes = log['active_model'] != log['active_model'].shift(1).fillna(method='bfill')
    chunk_masks = model_changes.cumsum()
    chunks = chunk_masks.unique()
    dataset_df = None
    for chunk in chunks:
        chunk_mask = chunk_masks == chunk
        all_state_active_similarity = log['all_state_active_similarity'].loc[chunk_mask].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
        if all_state_active_similarity.shape[0] < 1:
            continue
        active_model = log['active_model'].loc[chunk_mask].unique()[0]
        active_model_similarity = all_state_active_similarity[active_model]
        comparison_model_IDs = [m for m in all_state_active_similarity.columns if m != active_model]
        if len(comparison_model_IDs) < 1:
            continue
        comparison_similarity = all_state_active_similarity[comparison_model_IDs].mean(axis=1)
        chunk_df = log.loc[chunk_mask][['example', 'active_model']]
        chunk_df['active_similarity'] = active_model_similarity
        chunk_df['comparison_similarity'] = comparison_similarity
        chunk_df = chunk_df.dropna()
        if dataset_df is None:
            dataset_df = chunk_df
        else:
            dataset_df = dataset_df.append(chunk_df)
    if dataset_df is None:
        continue
    if dataset not in experiment_dfs:
        experiment_dfs[dataset] = {}
    if similarity_method not in experiment_dfs[dataset]:
        experiment_dfs[dataset][similarity_method] = dataset_df
    else:
        experiment_dfs[dataset][similarity_method] = experiment_dfs[dataset][similarity_method].append(dataset_df)

for dataset in experiment_dfs:
    for similarity_method in experiment_dfs[dataset]:
        print(dataset)
        print(similarity_method)
        print(experiment_dfs[dataset][similarity_method].reset_index().groupby('index').mean())
        plt.figure(figsize=(20,20))
        sns.lineplot(x='example', y='active_similarity', data=experiment_dfs[dataset][similarity_method].reset_index().groupby('index').mean())
        sns.lineplot(x='example', y='comparison_similarity', data=experiment_dfs[dataset][similarity_method].reset_index().groupby('index').mean())
        plt.savefig((output_path / f"{dataset}-{similarity_method}.pdf"))
        plt.clf()
        plt.close()




