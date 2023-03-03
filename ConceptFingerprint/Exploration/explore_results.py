import pickle
import json
import pathlib
import argparse

import numpy as np

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--name', default='Arabic', type=str, nargs='*')
# my_parser.add_argument('--cpu', default=2, type=int)

args = my_parser.parse_args()
names = args.name
data_type = "real"
data_source = "Real"

for name in names:
    output_path = pathlib.Path.cwd() / "output" / name

    results = list(output_path.glob('**/*results*'))

    margins = {}
    loss_by_featureset = {}
    sorted_loss = []
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
        # loss = result['all_distance_avg_loss']
        loss = (result['losses_by_distance_measure']['s_cosign'] + result['losses_by_distance_measure']['u_cosine']) / 2
        if np.isnan(loss):
            print("error")
        loss_by_featureset[str(metafeatures)] = loss
        sorted_loss.append((str(metafeatures), loss))
    print(loss_by_featureset)
    sorted_loss.sort(key=lambda x: x[1])

    metafeatures = {
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
    if str(metafeatures) not in loss_by_featureset:
        loss_by_featureset[str(metafeatures)] = 0.5
    for f in ['mean', 'stdev', 'skew', 'kurtosis', 'turning_point_rate', 'acf', 'pacf', 'MI', 'FI', 'IMF']:
        for k in metafeatures:
            metafeatures[k] = 1 if k == f or k[:-2] == f else 0
        if str(metafeatures) in loss_by_featureset:
            print(f"single featureset for {f}: {loss_by_featureset[str(metafeatures)]}")
    continue
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