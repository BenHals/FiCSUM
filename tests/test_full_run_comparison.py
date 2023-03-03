"""
Run a series of predefined options sets.
Compare results to saved results, to ensure no changes occur.
Test files are removed after running, but if stopped early can be found in ./testlog and ./testoutput
"""
import os
import sys
import shutil
import pandas as pd
import argparse
import pathlib
import json
import warnings
import random
import multiprocessing as mp

warnings.filterwarnings('ignore')

def get_results_fn(o_fn):
    return f"results_{o_fn.split('_options')[0]}.txt"
def get_command_fn(o_fn):
    return f"command_{o_fn.split('_options')[0]}.txt"
def get_data_fn(o_fn):
    return f"{o_fn.split('_options')[0]}.csv"


def run_task(co, condaenv):
    command_fn = co.parent / get_command_fn(co.stem)
    cmd = command_fn.open().read().replace('../run_experiment.py', str(pathlib.Path(__file__).absolute().parents[1] / 'run_experiment.py'))
    cmd = cmd.replace('../RawData', str(pathlib.Path(__file__).absolute().parents[1] / 'RawData'))
    cmd = cmd.replace('testoutput/', str(pathlib.Path(__file__).absolute().parent / 'testoutput/'))
    cmd = cmd.replace('testlog/', str(pathlib.Path(__file__).absolute().parent / 'testlog/'))
    print(cmd)
    if condaenv:
        os.system(f'conda activate {condaenv} && {cmd}')
    else:
        os.system(f'{cmd}')

def test_full_comparisons(condaenv=None, runall=False, cpu=1):
    testoutput_p = pathlib.Path(__file__).parent / './testoutput'
    comparison_p = pathlib.Path(__file__).parent / './comparisons'
    comparison_fns = list(comparison_p.rglob('*_options.txt'))

    options_exclude = ['base_output_path', 'raw_data_path', 'package_status', 'log_name', 'overall_time', 'overall_mem', 'peak_fingerprint_mem', 'average_fingerprint_mem', 'feature_weights', 'mean_discrimination']

    compare_to = comparison_fns if runall else random.choices(comparison_fns, k=2)

    if cpu == 1:
        for co in compare_to:
            run_task(co, condaenv)
    else:
        with mp.Pool(processes=cpu) as p:
            p.starmap(run_task, [(co, condaenv) for co in compare_to], chunksize=1)

    test_option_fns = list(testoutput_p.rglob('*_options.txt'))
    all_same = True
    for i,to in enumerate(test_option_fns):
        options = json.load(to.open())
        match_opts = None
        match_stem = None
        for co in comparison_fns:
            c_options = json.load(co.open())
            same = True
            for k in c_options:
                if k in options_exclude:
                    continue
                if c_options[k] != options[k]:
                    same = False
                    break
            if same:
                match_opts = c_options
                match_stem = co.stem
        test_results = json.load((to.parent / get_results_fn(to.stem)).open())
        compare_results = json.load((pathlib.Path(__file__).parent / './comparisons' / get_results_fn(match_stem)).open())

        same = True
        for k in compare_results:
            if k in options_exclude:
                continue
            if test_results[k] != compare_results[k]:
                same = False
                print(k)
                print(test_results[k], compare_results[k])
        if same:
            print(f"Test {i+1} Passed!")
        else:
            command_fn = (pathlib.Path(__file__).parent / './comparisons' / get_command_fn(match_stem)).open('r').read()
            print(f"Test {i+1} Failed! CMD was: {command_fn}")

        all_same = all_same & same

    shutil.rmtree(pathlib.Path(__file__).parent / './testlog')
    shutil.rmtree(pathlib.Path(__file__).parent / './testoutput')
    assert all_same


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--condaenv', default="streamFI", type=str)
    my_parser.add_argument('--cpus', default=1, type=int)
    my_parser.add_argument('--createtests', action='store_true')
    my_parser.add_argument('--runall', action='store_true')
    args = my_parser.parse_args()

    testoutput_p = pathlib.Path('./testoutput')
    comparison_p = pathlib.Path('./comparisons')
    comparison_fns = list(comparison_p.rglob('*_options.txt'))

    options_exclude = ['base_output_path', 'raw_data_path', 'package_status', 'log_name', 'overall_time', 'overall_mem', 'peak_fingerprint_mem', 'average_fingerprint_mem', 'feature_weights', 'mean_discrimination']




    # Generate new tests based on passed cmd option sets
    if args.createtests:
        cmd_strs = [
            'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --datalocation ../RawData --datasets cmc --outputlocation testoutput/ --loglocation testlog/ --single',
            'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --datalocation ../RawData --datasets STAGGERS --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI  --single',
            'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --datalocation ../RawData --datasets UCI-Wine --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI --isources features --single',
            'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --datalocation ../RawData --datasets STAGGERS --fingerprintmethod cachehistogram --fsmethod Cachehistogram_MI --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI --concept_max 3  --single',
            'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --datalocation ../RawData --datasets STAGGERS --fingerprintmethod cachehistogram --fsmethod Cachehistogram_MI --fingerprintbins 50 --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI --concept_max 3  --single',
            'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --datalocation ../RawData --datasets STAGGERS --fingerprintmethod cache --fsmethod CacheMIHy --fingerprintbins 25 --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI --concept_max 3  --single',
            'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --datalocation ../RawData --datasets STAGGERS --fingerprintmethod cachesketch --fsmethod sketch_MI --fingerprintbins 25 --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI --concept_max 3  --single',
            'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --datalocation ../RawData --datasets STAGGERS --fingerprintmethod cachesketch --fsmethod sketch_covMI --fingerprintbins 25 --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI --concept_max 3 --single',
            'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --datalocation ../RawData --datasets STAGGERS --outputlocation testoutput/ --loglocation testlog/ --single --ifeatures IMF MI --isources labels errors error_distances',
        ]

        for cmd_str in cmd_strs:
            os.system(f'conda activate {args.condaenv} && {cmd_str}')
            test_option_fn = list(testoutput_p.rglob('*_options.txt'))[0]
            cmd_fn = test_option_fn.parent / get_command_fn(test_option_fn.stem)
            rslt_fn = test_option_fn.parent / get_results_fn(test_option_fn.stem)
            data_fn = test_option_fn.parent / get_data_fn(test_option_fn.stem)
            with cmd_fn.open('w+') as f:
                f.write(cmd_str)

            shutil.move(src=str(test_option_fn), dst=str(pathlib.Path('./comparisons')))
            shutil.move(src=str(cmd_fn), dst=str(pathlib.Path('./comparisons')))
            shutil.move(src=str(rslt_fn), dst=str(pathlib.Path('./comparisons')))
            shutil.move(src=str(data_fn), dst=str(pathlib.Path('./comparisons')))

            shutil.rmtree('./testlog')
            shutil.rmtree('./testoutput')

    # Otherwise, test against the option sets in ./comparisons       
    else:
        test_full_comparisons(args.condaenv, args.runall, args.cpus)
