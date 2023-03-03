import pickle
import json
import pathlib
import argparse
import tqdm
from p_tqdm import p_map
from math import log
import multiprocessing as mp
from functools import partial


import numpy as np

from ConceptFingerprint.Exploration.explore_SW import run_eval, get_data
from ConceptFingerprint.Exploration.sliding_window_MI import learn_fingerprints_from_concepts, get_sliding_window_similarities, get_metafeature_name


def process_result_path(r_path, name, output_path, json, get_metafeature_name, get_data, run_eval):
    folder = r_path.parent
    option_path = folder / f"{name}_options.txt"
    with r_path.open() as f:
        result = json.load(f)
    with option_path.open() as f:
        options = json.load(f)
    run_path = output_path / get_metafeature_name(options)
    results_filename = run_path / f"{options['data_name'] }_results_redo.txt"
    if results_filename.exists():
        with results_filename.open() as f:
            result = json.load(f)
    else:
        stats, sims, real_concepts, length = get_data(run_path, name)
        losses, all_distance_avg_loss = run_eval(stats, sims, real_concepts, length, name, path=run_path, show=False, graph=False)
        result = {
            'losses_by_distance_measure': losses,
            'all_distance_avg_loss': all_distance_avg_loss
        }
        with open(str(results_filename), 'w') as f:
            json.dump(result, f)
    
    return [result, options, r_path]

def main():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--name', default='Arabic', type=str, nargs='*')
    my_parser.add_argument('--cpu', default=2, type=int)
    # my_parser.add_argument('--cpu', default=2, type=int)

    args = my_parser.parse_args()
    names = args.name
    data_type = "real"
    data_source = "Real"

    for name in names:
        output_path = pathlib.Path.cwd() / "output" / name

        results = list(output_path.glob('**/*results.txt'))

        margins = {}
        loss_by_featureset = {}
        sorted_loss = []
        process = lambda x: process_result_path(x, name, output_path)
        # pool = mp.Pool(processes=args.cpu)
        run_results = p_map(partial(process_result_path, name=name, output_path=output_path, json=json, get_metafeature_name=get_metafeature_name, get_data=get_data, run_eval=run_eval), results, num_cpus=args.cpu)
        for result, options, r_path in tqdm.tqdm(run_results):
            # result = process_result_path(r_path)
            # print(result)
            # print(options)
            metafeatures = options['meta-features']
            # loss = result['all_distance_avg_loss']
            loss = (result['losses_by_distance_measure']['s_cosign'] + result['losses_by_distance_measure']['u_cosine']) / 2
            if np.isnan(loss):
                print("error")
            loss_by_featureset[str(metafeatures)] = loss
            sorted_loss.append((str(metafeatures), loss))
        print(loss_by_featureset)
        sorted_loss.sort(key=lambda x: x[1])

        metafeatures_null = {
                "mean": 0,
                "stdev": 0,
                "skew": 0,
                "kurtosis": 0,
                "turning_point_rate": 0,
                "acf_1": 0,
                "acf_2": 0,
                "pacf_1": 0,
                "pacf_2": 0,
                "MI": 0,
                "FI": 0,
                "IMF_0": 0,
                "IMF_1": 0,
                "IMF_2": 0,
            }
        if str(metafeatures_null) not in loss_by_featureset:
            loss_by_featureset[str(metafeatures_null)] = -log(0.5)
        for f in ['mean', 'stdev', 'skew', 'kurtosis', 'turning_point_rate', 'acf', 'pacf', 'MI', 'FI', 'IMF']:
            for k in metafeatures:
                metafeatures[k] = 1 if k == f or k[:-2] == f else 0
            if str(metafeatures) in loss_by_featureset:
                print(f"single featureset for {f} > {str(metafeatures)}: {loss_by_featureset[str(metafeatures)]} diff: {loss_by_featureset[str(metafeatures)] - loss_by_featureset[str(metafeatures_null)] }")
        # continue
        for r_path in results:
            folder = r_path.parent
            option_path = folder / f"{name}_options.txt"
            with r_path.open() as f:
                result = json.load(f)
            with option_path.open() as f:
                options = json.load(f)
            
            print(result)
            print(options)
            metafeatures = options['meta-features']
            MF_options = ['mean', 'stdev', 'skew', 'kurtosis', 'turning_point_rate', 'acf', 'pacf', 'MI', 'FI', 'IMF']
            for feature in MF_options:
                in_options = False
                for k in metafeatures:
                    if k[:-2] == feature or k == feature:
                        in_options = True
                if in_options:
                    marginal_featureset = {**metafeatures}
                    print('****')
                    print(metafeatures)
                    print(loss_by_featureset[str(metafeatures)])
                    for k in marginal_featureset:
                        if k == feature or k[:-2] == feature:
                            marginal_featureset[k] = 0
                    print(marginal_featureset)
                    if str(marginal_featureset) in loss_by_featureset:
                        print(f"marginal_featureset in dataset")
                        print(loss_by_featureset[str(marginal_featureset)])
                        if feature not in margins:
                            margins[feature] = []
                        loss_delta = loss_by_featureset[str(metafeatures)] - loss_by_featureset[str(marginal_featureset)]
                        loss_delta_percent = loss_delta / loss_by_featureset[str(metafeatures)]
                        # if np.isnan(loss_delta):
                        #     print("error")
                        print(f"Marginal loss diff for {feature} is {loss_delta}")
                        print(f"Marginal loss percent for {feature} is {loss_delta_percent}")
                        margins[feature].append(loss_delta_percent)

    for k in margins.keys():
        # print(margins[k])
        print(f"{k} : {np.mean(margins[k])}")

    print(sorted_loss[:5])
    # print(margins)

if __name__ == "__main__":
    main()