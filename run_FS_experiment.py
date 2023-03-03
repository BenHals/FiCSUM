import argparse
import warnings
from collections import deque, Counter
import tqdm
import json
import pathlib
import pickle
import copy
import math, itertools
import logging, time
from logging.handlers import RotatingFileHandler
import subprocess, os, csv
import psutil, time
import multiprocessing as mp
from multiprocessing import freeze_support, RLock

import numpy as np
import matplotlib.pyplot as plt

from ConceptFingerprint.Data.load_data import \
        load_synthetic_concepts,\
        load_real_concepts,\
        get_inorder_concept_ranges,\
        AbruptDriftStream

from ConceptFingerprint.Classifier.meta_info_classifier import DSClassifier
from ConceptFingerprint.Classifier.hoeffding_tree_shap import HoeffdingTreeSHAPClassifier
import importlib
import pandas as pd
from pyinstrument import Profiler
from ConceptFingerprint.asizeof import asizeof, asized
from pympler.classtracker import ClassTracker


class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None
        
        elif isinstance(obj, pathlib.Path):
            return str(obj)

        return json.JSONEncoder.default(self, obj)

# Load synthetic concepts (up to concept_max) and turn into a stream
# Concepts are placed in order (repeats) number of times.
# The idea is to use half the stream to learn concepts, then half
# to test for similarity, so 2 repeats to learn concepts then
# test on 2.
def make_stream(options):
    if options['data_type'] == "Synthetic":
        concepts = load_synthetic_concepts(options['data_name'],
                                            options['seed'],
                                            raw_data_path = options['raw_data_path'] / 'Synthetic')
    else:
        concepts = load_real_concepts(options['data_name'],
                                        options['seed'],
                                        nrows = options['max_rows'],
                                        raw_data_path = options['raw_data_path'] / 'Real',
                                        sort_examples=True)
    # for concept_gen, _ in concepts:
        # concept_gen.prepare_for_use()
    stream_concepts, length = get_inorder_concept_ranges(concepts, concept_length=options['concept_length'], seed=options['seed'], repeats=options['repeats'], concept_max=options['concept_max'], repeat_proportion=options['repeatproportion'])
    options['length'] = length
    try:
        stream = AbruptDriftStream(stream_concepts, length)
    except:
        return None, None, None, None
    # stream.prepare_for_use()
    all_classes = stream._get_target_values()
    return stream, stream_concepts, length, list(all_classes)

def get_package_status():
    data = []
    for package in ["ConceptFingerprint"]:
        try:
            loc = str(importlib.util.find_spec(package).submodule_search_locations[0])
        except Exception as e:
            try:
                loc = str(importlib.util.find_spec(package).submodule_search_locations._path[0])
            except:
                namespace = importlib.util.find_spec(package).submodule_search_locations
                loc = str(namespace).split('[')[1].split(']')[0]
                loc = loc.split(',')[0]
                loc = loc.replace("'", "")
        loc = str(pathlib.Path(loc).resolve())
        commit = subprocess.check_output(["git", "describe", "--always"], cwd=loc).strip().decode()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=loc).strip().decode()
        # changes = subprocess.check_output(["git", "diff", "--quiet", "&&", "git", "diff", "--cached", "--quiet"], cwd=loc).strip().decode()
        changes = subprocess.call(["git", "diff", "--quiet"], cwd=loc)
        changes_cached = subprocess.call(["git", "diff", "--cached", "--quiet"])
        if changes:
            print(f"{package} has uncommitted files: {changes}")
            input("Are you sure you want to run with uncommitted code? Press any button to continue...")
        package_data = f"{package}-{branch}-{commit}"
        data.append(package_data)
    return '_'.join(data)

def process_option(option):
    np.seterr(all='raise')
    warnings.filterwarnings('error')
    mp_process = mp.current_process()
    mp_id = 1
    try:
        mp_id = int(mp_process.name.split('-')[1])
    except:
        pass
    # for i in tqdm.tqdm(range(100), position=mp_id+1, desc=f"Worker {mp_id}", leave=False):
    #     time.sleep(0.1)
    # return 1
    def get_drift_point_accuracy(log, follow_length = 250):
        if not 'drift_occured' in log.columns or not 'is_correct' in log.columns:
            return 0, 0, 0, 0
        dpl = log.index[log['drift_occured'] == 1].tolist() 
        dpl = dpl[1:]
        if len(dpl) == 0:
            return 0, 0, 0, 0

        following_drift = np.unique(np.concatenate([np.arange(i, min(i+follow_length+1, len(log))) for i in dpl]))
        filtered = log.iloc[following_drift]
        num_close = filtered.shape[0]
        accuracy, kappa, kappa_m, kappa_t = get_performance(filtered)
        return accuracy, kappa, kappa_m, kappa_t

    def get_driftdetect_point_accuracy(log, follow_length = 250):
        if not 'change_detected' in log.columns:
            return 0, 0, 0, 0
        if not 'drift_occured' in log.columns:
            return 0, 0, 0, 0
        dpl = log.index[log['change_detected'] == 1].tolist()   
        drift_indexes = log.index[log['drift_occured'] == 1].tolist()   
        if len(dpl) < 1:
            return 0, 0, 0, 0
        following_drift = np.unique(np.concatenate([np.arange(i, min(i+1000+1, len(log))) for i in drift_indexes]))
        following_detect= np.unique(np.concatenate([np.arange(i, min(i+follow_length+1, len(log))) for i in dpl]))
        following_both = np.intersect1d(following_detect, following_drift, assume_unique= True)
        filtered = log.iloc[following_both]
        num_close = filtered.shape[0]
        if num_close == 0:
            return 0, 0, 0, 0
        accuracy, kappa, kappa_m, kappa_t = get_performance(filtered)
        return accuracy, kappa, kappa_m, kappa_t
    
    def get_performance(log):
        sum_correct = log['is_correct'].sum()
        num_observations = log.shape[0]
        accuracy = sum_correct / num_observations
        values, counts = np.unique(log['y'], return_counts = True)
        majority_class = values[np.argmax(counts)]
        majority_correct = log.loc[log['y'] == majority_class]
        num_majority_correct = majority_correct.shape[0]
        majority_accuracy =  num_majority_correct / num_observations
        kappa_m = (accuracy - majority_accuracy) / (1 - majority_accuracy)
        temporal_filtered = log['y'].shift(1, fill_value = 0.0)
        temporal_correct = log['y'] == temporal_filtered
        temporal_accuracy = temporal_correct.sum() / num_observations
        kappa_t = (accuracy - temporal_accuracy) / (1 - temporal_accuracy)

        our_counts = Counter()
        gt_counts = Counter()
        for v in values:
            our_counts[v] = log.loc[log['p'] == v].shape[0]
            gt_counts[v] = log.loc[log['y'] == v].shape[0]
        
        expected_accuracy = 0
        for cat in values:
            expected_accuracy += (gt_counts[cat] * our_counts[cat]) / num_observations
        expected_accuracy /= num_observations
        kappa = (accuracy - expected_accuracy) / (1 - expected_accuracy)


        return accuracy, kappa, kappa_m, kappa_t

    def get_recall_precision(log):
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
            'GT_to_MODEL_ratio':0,
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
        return overall_results

    def get_discrimination_results(log):
        """ Calculate how many standard deviations the active state
        is from other states. 
        We first split the active state history into chunks representing 
        each segment.
        We then shrink this by 50 on each side to exclude transition periods.
        We then compare the distance from the active state to each non-active state
        in terms of stdev. We use the max of the active state stdev or comparison stdev
        for the given chunk, representing how much the active state could be discriminated
        from the comparison state.
        We return a set of all comparisons, a set of average per active state, and overall average.
        """
        models = log['active_model'].unique()
        all_state_active_similarity = log['all_state_active_similarity'].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
        if len(all_state_active_similarity.columns) == 0:
            return -1, None, None 
        # Scale to between 0 and 1, so invariant
        # to the size of the similarity function.
        # max_similarity = None
        # min_similarity = None
        # for m in all_state_active_similarity.columns:
        #     max_val = all_state_active_similarity[m].max()
        #     min_val = all_state_active_similarity[m].min()
        #     max_similarity = max(max_similarity, max_val) if max_similarity is not None else max_val
        #     min_similarity = min(min_similarity, min_val) if min_similarity is not None else min_val

        values = np.concatenate([all_state_active_similarity[m].dropna().values for m in  all_state_active_similarity.columns])
        max_similarity = np.percentile(values, 90)
        min_similarity = min(values)

        # Split into chunks using the active model.
        # I.E. new chunk every time the active model changes.
        # We shrink chunks by 50 each side to discard transition.
        model_changes = log['active_model'] != log['active_model'].shift(1).fillna(method='bfill')
        chunk_masks = model_changes.cumsum()
        chunks = chunk_masks.unique()
        divergences = {}
        active_model_mean_divergences = {}
        mean_divergence = []

        # Find the number of observations we are interested in.
        # by combining chunk masks.
        all_chunks = None
        for chunk in chunks:
            chunk_mask = chunk_masks == chunk
            chunk_shift = chunk_mask.shift(50, fill_value=0)
            smaller_mask = chunk_mask & chunk_shift
            chunk_shift = chunk_mask.shift(-50, fill_value=0)
            smaller_mask = smaller_mask & chunk_shift
            all_state_active_similarity = log['all_state_active_similarity'].loc[smaller_mask].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
            
            # We skip chunks with only an active state.
            if len(all_state_active_similarity.columns) < 2:
                continue
            if all_chunks is None:
                all_chunks = smaller_mask
            else:
                all_chunks = all_chunks | smaller_mask
        
        # If we only have one state, we don't
        # have any divergences
        if all_chunks is None:
            return None, None, 0

        for chunk in chunks:
            chunk_mask = chunk_masks == chunk
            chunk_shift = chunk_mask.shift(50, fill_value=0)
            smaller_mask = chunk_mask & chunk_shift
            chunk_shift = chunk_mask.shift(-50, fill_value=0)
            smaller_mask = smaller_mask & chunk_shift

            # state similarity is saved in the csv as a ; seperated list, where the index is the model ID.
            # This splits this column out into a column per model.
            all_state_active_similarity = log['all_state_active_similarity'].loc[smaller_mask].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
            if all_state_active_similarity.shape[0] < 1:
                continue
            active_model = log['active_model'].loc[smaller_mask].unique()[0]
            for m in all_state_active_similarity.columns:
                all_state_active_similarity[m] = (all_state_active_similarity[m] - min_similarity) / (max_similarity - min_similarity)
                all_state_active_similarity[m] = np.clip(all_state_active_similarity[m], 0, 1)
            # Find the proportion this chunk takes up of the total.
            # We use this to proportion the results.
            chunk_proportion = smaller_mask.sum() / all_chunks.sum()
            chunk_mean = []
            for m in all_state_active_similarity.columns:
                if m == active_model:
                    continue
                
                # If chunk is small, we may only see 0 or 1 observations.
                # We can't get a standard deviation from this, so we skip.
                if all_state_active_similarity[m].shape[0] < 2:
                    continue
                # Use the max of the active state, and comparison state as the Stdev.
                # You cannot distinguish if either is larger than difference.
                scale = np.mean([all_state_active_similarity[m].std(), all_state_active_similarity[active_model].std()])
                divergence = all_state_active_similarity[m] - all_state_active_similarity[active_model]
                avg_divergence = divergence.sum() / divergence.shape[0]

                scaled_avg_divergence = avg_divergence / scale if scale > 0 else 0

                # Mutiply by chunk proportion to average across data set.
                # Chunks are not the same size, so cannot just mean across chunks.
                scaled_avg_divergence *= chunk_proportion
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

        # Use sum because we multiplied by proportion already, so just need to add up.
        mean_divergence = np.sum(mean_divergence)
        for m in active_model_mean_divergences:
            active_model_mean_divergences[m] = np.sum(active_model_mean_divergences[m])
        
        return divergences, active_model_mean_divergences, mean_divergence
    
    def plot_feature_weights(log):
        feature_weights = log['feature_weights'].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True)


    def dump_results(option, log_path, result_path, log=None):
        log_df = None
        if log is not None:
            log_df = log
        else:
            log_df = pd.read_csv(log_dump_path)
        overall_accuracy = log_df['overall_accuracy'].values[-1]
        overall_time = log_df['cpu_time'].values[-1]
        overall_mem = log_df['ram_use'].values[-1]
        peak_fingerprint_mem = log_df['ram_use'].values.max()
        average_fingerprint_mem = log_df['ram_use'].values.mean()

        acc, kappa, kappa_m, kappa_t = get_performance(log_df)
        result = {
            'overall_accuracy': overall_accuracy,
            'acc': acc,
            'kappa': kappa,
            'kappa_m': kappa_m,
            'kappa_t': kappa_t,
            'overall_time': overall_time,
            'overall_mem': overall_mem,
            'peak_fingerprint_mem': peak_fingerprint_mem,
            'average_fingerprint_mem': average_fingerprint_mem,
            **option
            }
        for delta in [50, 250, 500]:
            acc, kappa, kappa_m, kappa_t = get_drift_point_accuracy(log_df, delta)
            result[f"drift_{delta}_accuracy"] = acc
            result[f"drift_{delta}_kappa"] = kappa
            result[f"drift_{delta}_kappa_m"] = kappa_m
            result[f"drift_{delta}_kappa_t"] = kappa_t
            acc, kappa, kappa_m, kappa_t = get_driftdetect_point_accuracy(log_df, delta)
            result[f"driftdetect_{delta}_accuracy"] = acc
            result[f"driftdetect_{delta}_kappa"] = kappa
            result[f"driftdetect_{delta}_kappa_m"] = kappa_m
            result[f"driftdetect_{delta}_kappa_t"] = kappa_t

        match_results = get_recall_precision(log_df)
        for k,v in match_results.items():
            result[k] = v

        discriminations, active_model_mean_discriminations, mean_discrimination = get_discrimination_results(log_df)
        result['mean_discrimination'] = mean_discrimination

        with result_path.open('w+') as f:
            json.dump(result, f, cls=NpEncoder)

    profiler = None
    if option['pyinstrument']:
        profiler = Profiler()
        profiler.start()
    stream, stream_concepts, length, classes = make_stream(option)
    if stream is None:
        return None
    # name = f"{option['similarity_option'][0]}-{'_'.join([x[0] for x in option['isources']])}-{'_'.join([x[0] for x in option['ifeatures']])}-{str(option['window_size'])[0]}-{str(option['similarity_gap'])[0]}-{str(option['fp_gap'])[0]}-{str(option['nonactive_fp_gap'])[0]}"
    UID = hash(tuple(f"{k}{str(v)}" for k,v in option.items()))
    window_size = option['window_size']

    learner = HoeffdingTreeSHAPClassifier

    #use an observation_gap of -1 as auto, take 1000 observations across the stream
    if option['observation_gap'] == -1:
        option['observation_gap'] = math.floor(length / 1000)

    classifier = DSClassifier(
                learner = learner,
                window_size = option['window_size'],
                similarity_gap = option['similarity_gap'],
                fingerprint_update_gap = option['fp_gap'],
                non_active_fingerprint_update_gap = option['nonactive_fp_gap'],
                observation_gap = option['observation_gap'],
                sim_measure = option['similarity_option'],
                ignore_sources = option['isources'],
                ignore_features = option['ifeatures'],
                similarity_num_stdevs = option['similarity_stdev_thresh'],
                similarity_min_stdev = option['similarity_stdev_min'],
                similarity_max_stdev = option['similarity_stdev_max'],
                buffer_ratio=option['buffer_ratio'],
                feature_selection_method=option['fs_method'],
                fingerprint_method=option['fingerprint_method'],
                fingerprint_bins=option['fingerprint_bins'],)
    
    output_path = option['base_output_path'] / option['experiment_name'] / option['package_status'] / option['data_name'] / str(option['seed'])
    output_path.mkdir(parents=True, exist_ok=True)
    run_index = 0
    run_name = f"run_{UID}_{run_index}"
    log_dump_path = output_path / f"{run_name}.csv"
    options_dump_path = output_path / f"{run_name}_options.txt"
    options_dump_path_partial = output_path / f"partial_{run_name}_options.txt"
    results_dump_path = output_path / f"results_{run_name}.txt"

    # Look for existing file using this name.
    # This will happen for all other option types, so
    # look for other runs with the same options.
    json_options = json.loads(json.dumps(option, cls=NpEncoder))
    def compare_options(A, B):
        for k in A:
            if k in ['log_name', 'package_status', 'base_output_path']:
                continue
            if k not in A or k not in B:
                continue
            if A[k] != B[k]:
                return False
        return True
    other_runs = output_path.glob('*_options.txt')
    for other_run_path in other_runs:
        if 'partial' in other_run_path.stem:
            continue
        else:
            with other_run_path.open() as f:
                existing_options = json.load(f)
            if compare_options(existing_options, json_options):
                if option['pyinstrument']:
                    profiler.stop()
                return other_run_path
                    
    while options_dump_path.exists() or options_dump_path_partial.exists():
        run_index += 1
        run_name = f"runother_{UID}_{run_index}"
        log_dump_path = output_path / f"{run_name}.csv"
        options_dump_path = output_path / f"{run_name}_options.txt"
        options_dump_path_partial = output_path / f"partial_{run_name}_options.txt"
        results_dump_path = output_path / f"results_{run_name}.txt"
    
        



        
    
    partial_log_size = 2500
    partial_logs = []
    partial_log_index = 0


    with options_dump_path_partial.open('w+') as f:
        json.dump(option, f, cls=NpEncoder)

    

    right = 0
    wrong = 0
    stream_names = [c[3] for c in stream_concepts]


    monitoring_data = []
    monitoring_header = ('example', 'y', 'p', 'is_correct', 'right_sum', 'wrong_sum', 'overall_accuracy', 'active_model', 'ground_truth_concept', 'drift_occured', 'change_detected', 'model_evolution', 'active_state_active_similarity', 'active_state_buffered_similarity', 'all_state_buffered_similarity', 'all_state_active_similarity', 'feature_weights', 'cpu_time', 'ram_use', 'fingerprint_ram')
    logging.info(option)
    start_time = time.process_time()
    
    def memory_usage_psutil():
        # return the memory usage in MB
        process = psutil.Process(os.getpid())
        mem = process.memory_info()[0] / float(2 ** 20)
        return mem
    start_mem = memory_usage_psutil()
    tracker = ClassTracker()
    tracker.track_class(classifier.fingerprint_type)
    tracker.create_snapshot()
    for i in tqdm.tqdm(range(option['length']), position=mp_id+1, desc=f"Worker {mp_id} {str(UID)[:4]}...", leave=False):
        observation_monitoring = {}
        observation_monitoring['example'] = i
        X, y = stream.next_sample()
        # TODO: Remove testing
        #<testing>
        for f in X:
            if 1.0 in f or 0.0 in f:
                print("weird")
        #<\testing>
        
        observation_monitoring['y'] = y[0]
        p = classifier.predict(X)
        observation_monitoring['p'] = y[0]
        e = y[0] == p[0]
        observation_monitoring['is_correct'] = e
        right += y[0] == p[0]
        observation_monitoring['right_sum'] = right
        wrong += y[0] != p[0]
        observation_monitoring['wrong_sum'] = wrong
        observation_monitoring['overall_accuracy'] = right / (right + wrong)

        # Find ground truth active concept
        ground_truth_concept_index = None
        for c in stream_concepts:
            concept_start= c[0]
            if concept_start <= i < c[1]:
                ground_truth_concept_index = stream_names.index(c[3])

        # Control parameters
        drift_occured = False
        concept_drift = False
        concept_drift_target = None
        concept_transition = False
        classifier.manual_control = False
        classifier.force_stop_learn_fingerprint = False
        classifier.force_transition = False
        if option['optdetect']:
            classifier.force_transition_only = True
        
        
        # For control, find if there was a ground truth
        # drift, with some delay.
        for c in stream_concepts[1:]:
            concept_start= c[0]
            if i == concept_start:
                drift_occured = True
        for c in stream_concepts[1:]:
            concept_start= c[0]
            if i == concept_start + window_size + 10:
                concept_drift = True
                concept_drift_target = stream_names.index(c[3])
            # if concept_start <= i < concept_start + 275:
            #     concept_transition = True
        if concept_drift and option['optdetect']:
            classifier.manual_control = True
            classifier.force_transition = True
            classifier.force_stop_learn_fingerprint = True
            if option['optselect']:
                classifier.force_transition_to = concept_drift_target
        if concept_transition:
            classifier.force_stop_learn_fingerprint = True
        
        # classifier.force_stop_fingerprint_age = 2999
        # if i >= 4000:
        #     classifier.force_lock_weights = True
        #     classifier.force_stop_add_to_normalizer = True
        classifier.partial_fit(X, y, classes=classes)
        # Collect monitoring data for storage.
        current_active_model = classifier.active_state
        observation_monitoring['active_model'] = current_active_model
        observation_monitoring['ground_truth_concept'] = ground_truth_concept_index
        observation_monitoring['drift_occured'] = drift_occured
        observation_monitoring['change_detected'] = classifier.detected_drift
        observation_monitoring['model_evolution'] = classifier.get_active_state().current_evolution

        if classifier.monitor_active_state_active_similarity is not None:
            observation_monitoring['active_state_active_similarity'] = classifier.monitor_active_state_active_similarity
        else:
            observation_monitoring['active_state_active_similarity'] = -1

        if classifier.monitor_active_state_buffered_similarity is not None:
            observation_monitoring['active_state_buffered_similarity'] = classifier.monitor_active_state_buffered_similarity
        else:
            observation_monitoring['active_state_buffered_similarity'] = -1

        buffered_data = classifier.monitor_all_state_buffered_similarity
        if buffered_data is not None:
            buffered_accuracy, buffered_stats, buffered_window, buffered_similarities = buffered_data
            concept_similarities = [(int(k), v) for k,v in buffered_similarities.items() if k != 'active']
            concept_similarities.sort(key = lambda x: x[0])
            observation_monitoring['all_state_buffered_similarity'] = ';'.join([str(x[1]) for x in concept_similarities])
        else:
            observation_monitoring['all_state_buffered_similarity'] = -1
        
        weights = classifier.monitor_feature_selection_weights
        if weights is not None:
            observation_monitoring['feature_weights'] = ';'.join([f"{s}{f}:{v}" for s,f,v in weights])
        else:
            observation_monitoring['feature_weights'] = -1

        all_state_active_similarity = classifier.monitor_all_state_active_similarity
        if all_state_active_similarity is not None:
            active_accuracy, active_stats, active_fingerprints, active_window, active_similarities = all_state_active_similarity
            concept_similarities = [(int(k), v) for k,v in active_similarities.items() if k != 'active']
            concept_similarities.sort(key = lambda x: x[0])
            observation_monitoring['all_state_active_similarity'] = ';'.join([str(x[1]) for x in concept_similarities])
        else:
            observation_monitoring['all_state_active_similarity'] = -1
        
        observation_monitoring['detected_drift'] = classifier.detected_drift
        observation_monitoring['concept_drift'] = concept_drift

        observation_monitoring['cpu_time'] = time.process_time() - start_time
        observation_monitoring['ram_use'] = memory_usage_psutil() - start_mem
        last_memory_snap = tracker.snapshots[-1]
        if hasattr(last_memory_snap, "classes"):
            observation_monitoring['fingerprint_ram'] = last_memory_snap.classes[-1]['avg']
        else:
            observation_monitoring['fingerprint_ram'] = 0.0

        monitoring_data.append(observation_monitoring)

        dump_start = time.process_time()
        if len(monitoring_data) >= partial_log_size:
            tracker.create_snapshot()
            # tracker.stats.print_summary()
            # non_active_size = asizeof([x.non_active_fingerprints for x in classifier.states.values()]) / (1024 * 1024)
            # active_size = asizeof([x.fingerprint for x in classifier.states.values()]) / (1024 * 1024)
            # print(non_active_size)
            # print(active_size)
            # print(non_active_size + active_size)
            log_dump_path_partial = output_path / f"partial_{run_name}_{partial_log_index}.csv"
            df = pd.DataFrame(monitoring_data, columns=monitoring_header)
            df.to_csv(log_dump_path_partial)
            partial_log_index += 1
            partial_logs.append(log_dump_path_partial)
            monitoring_data = []
            df = None
        dump_end = time.process_time()
        dump_time = dump_end - dump_start
        start_time -= dump_time
    df = None
    for partial_log in partial_logs:
        if df is None:
            df = pd.read_csv(partial_log)
        else:
            next_log = pd.read_csv(partial_log)
            df = df.append(next_log)
    if df is None:
        df = pd.DataFrame(monitoring_data, columns=monitoring_header)
    else:
        df = df.append(pd.DataFrame(monitoring_data, columns=monitoring_header))
    df = df.reset_index()
    df.to_csv(log_dump_path)
    with options_dump_path.open('w+') as f:
        json.dump(option, f, cls=NpEncoder)
    
    for partial_log in partial_logs:
        partial_log.unlink()
    options_dump_path_partial.unlink()

    dump_results(option, log_dump_path, results_dump_path, df)
    if option['pyinstrument']:
        profiler.stop()
        res = profiler.output_text(unicode=True, color=True)
        print(res)
        with open("profile.txt", 'w+') as f:
            f.write(res)
    

    return options_dump_path


            
        

# logging.basicConfig(level=logging.INFO, filename=f"logs/experiment-{time.time()}.log", filemode='w')
if __name__ == "__main__":
    freeze_support()
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--datasets', default='cmc', nargs="*", type=str)
    my_parser.add_argument('--seeds', default=1, nargs="*", type=int)
    my_parser.add_argument('--seedaction', default="list", type=str, choices=['new', 'list', 'reuse'])
    my_parser.add_argument('--datalocation', default="RawData", type=str)
    my_parser.add_argument('--outputlocation', default="output", type=str)
    my_parser.add_argument('--loglocation', default="experimentlog", type=str)
    my_parser.add_argument('--experimentname', default="expDefault", type=str)
    my_parser.add_argument('--optionslocation', default=None, type=str)
    my_parser.add_argument('--fsmethod', default="default", type=str, nargs="*")
    my_parser.add_argument('--fingerprintmethod', default="descriptive", type=str, nargs="*")
    my_parser.add_argument('--fingerprintbins', default=10, type=int, nargs="*")
    my_parser.add_argument('--logging', action='store_true')
    my_parser.add_argument('--pyinstrument', action='store_true')
    my_parser.add_argument('--single', action='store_true')
    my_parser.add_argument('--experimentopts', action='store_true')
    my_parser.add_argument('--cpu', default=2, type=int)
    my_parser.add_argument('--repeats', default=3, type=int)
    my_parser.add_argument('--concept_length', default=5000, type=int)
    my_parser.add_argument('--concept_max', default=6, type=int)
    my_parser.add_argument('--repeatproportion', default=0.75, type=float)
    my_parser.add_argument('--maxrows', default=75000, type=int)
    my_parser.add_argument('--sim', default='metainfo', nargs="*", type=str)
    my_parser.add_argument('--window_size', default=75, nargs="*", type=int)
    my_parser.add_argument('--sim_gap', default=3, nargs="*", type=int)
    my_parser.add_argument('--fp_gap', default=50, nargs="*", type=int)
    my_parser.add_argument('--na_fp_gap', default=100, nargs="*", type=int)
    my_parser.add_argument('--ob_gap', default=-1, nargs="*", type=int)
    my_parser.add_argument('--sim_stdevs', default=3, nargs="*", type=float)
    my_parser.add_argument('--min_sim_stdev', default=0.01, nargs="*", type=float)
    my_parser.add_argument('--max_sim_stdev', default=0.1, nargs="*", type=float)
    my_parser.add_argument('--buffer_ratio', default=0.25, nargs="*", type=float)
    my_parser.add_argument('--optdetect', action='store_true')
    my_parser.add_argument('--optselect', action='store_true')
    my_parser.add_argument('--isources', nargs="*", help="set sources to be ignored, from feature, f{i}, labels, predictions, errors, error_distances")
    my_parser.add_argument('--ifeatures', default=['IMF', 'MI'], nargs="*", help="set features to be ignored, any meta-information feature")
    my_parser.add_argument('--classes', nargs='*', help='We try to detect classes automatically\
                                when the normalizer is set up, but sometimes this does not find\
                                rare classes. In this case, manually pass all clases in the dataset.')
    args = my_parser.parse_args()

    real_drift_datasets = ['AQSex', 'AQTemp', 'Arabic', 'cmc', 'UCI-Wine', 'qg']
    real_unknown_datasets = ['Airlines', 'Arrowtown', 'AWS', 'Beijing', 'covtype', 'Elec', 'gassensor', 'INSECTS-abn', 'INSECTS-irbn', 'INSECTS-oocn', 'KDDCup', 'Luxembourg', 'NOAA', 'outdoor', 'ozone', 'Poker', 'PowerSupply', 'Rangiora', 'rialto', 'Sensor', 'Spam', 'SpamAssassin']
    synthetic_MI_datasets = ['RTREESAMPLE', 'HPLANESAMPLE']
    # synthetic_perf_only_datasets = ['RTREE', 'HPLANE', 'STAGGER', 'RTREEEasy', 'RTREEEasySAMPLE', 'RTREEMedSAMPLE', 'RBFEasy', 'RTREEEasyF', 'RBFMed']
    synthetic_perf_only_datasets = ['STAGGER', 'RTREEMedSAMPLE', 'RBFMed']
    synthetic_unused = ['STAGGERS', 'RTREE', 'HPLANE', 'RTREEEasy', 'RTREEEasySAMPLE', 'RBFEasy', 'RTREEEasyF', 'RTREEEasyA', 'SynEasyD', 'SynEasyA', 'SynEasyF', 'SynEasyDA', 'SynEasyDF', 'SynEasyAF', 'SynEasyDAF']
    synthetic_dist = ['RTREESAMPLE-UB', 'RTREESAMPLE-NB', 'RTREESAMPLE-DB', 'RTREESAMPLE-UU', 'RTREESAMPLE-UN', 'RTREESAMPLE-UD', 'RTREESAMPLE-NU', 'RTREESAMPLE-NN', 'RTREESAMPLE-ND', 'RTREESAMPLE-DU', 'RTREESAMPLE-DN', 'RTREESAMPLE-DD']
    datasets = set()
    for ds in (args.datasets if type(args.datasets) is list else [args.datasets]):
        if ds == 'all_exp':
            for x in [*real_drift_datasets]:
                datasets.add((x, 'Real'))
            for x in [*synthetic_MI_datasets, *synthetic_perf_only_datasets]:
                datasets.add((x, 'Synthetic'))
        elif ds == 'real':
            for x in [*real_drift_datasets]:
                datasets.add((x, 'Real'))
        elif ds == 'synthetic':
            for x in [*synthetic_MI_datasets, *synthetic_perf_only_datasets, *synthetic_dist]:
                datasets.add((x, 'Synthetic'))
        elif ds in real_drift_datasets:
            datasets.add((ds, 'Real'))
        elif ds in real_unknown_datasets:
            datasets.add((ds, 'Real'))
        elif ds in synthetic_MI_datasets:
            datasets.add((ds, 'Synthetic'))
        elif ds in synthetic_perf_only_datasets:
            datasets.add((ds, 'Synthetic'))
        elif ds in synthetic_unused:
            datasets.add((ds, 'Synthetic'))
        elif ds in synthetic_dist:
            datasets.add((ds, 'Synthetic'))
        else:
            raise ValueError("Dataset not found")

    seeds = []
    num_seeds = 0
    base_seeds = []
    if args.seedaction == 'reuse':
        num_seeds = args.seeds if type(args.seeds) is not list else args.seeds[0]
        base_seeds = np.random.randint(0, 9999, size=num_seeds)
    if args.seedaction == 'new':
        num_seeds = args.seeds if type(args.seeds) is not list else args.seeds[0]
        seeds = np.random.randint(0, 9999, size=num_seeds)
    if args.seedaction == 'list':
        seeds = args.seeds if type(args.seeds) is list else [args.seeds]
        num_seeds = len(seeds)

    raw_data_path = pathlib.Path(args.datalocation).resolve()
    if not raw_data_path.exists():
        raise ValueError(f"Data location {raw_data_path} does not exist")
    for ds, ds_type in datasets:
        data_file_location = raw_data_path / ds_type / ds
        if not data_file_location.exists():
            if ds_type == 'Synthetic':
                data_file_location.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(f"Data file {data_file_location} does not exist")
    base_output_path = pathlib.Path(args.outputlocation).resolve()
    if not base_output_path.exists():
        base_output_path.mkdir(parents=True)
    log_path = pathlib.Path(args.loglocation).resolve()
    if not log_path.exists():
        log_path.mkdir(parents=True)
    given_options = None
    if args.optionslocation is not None:
        options_path = pathlib.Path(args.optionslocation).resolve()
        if options_path.exists():
            with options_path.open() as f:
                given_options = json.load(f)

    log_name = f"{args.experimentname}-{time.time()}"
    if args.logging:
        logging.basicConfig(handlers=[RotatingFileHandler(f"{log_path}/{log_name}.log", maxBytes=500000000, backupCount=100)],level=logging.INFO)
    with (log_path / f"e{log_name}.txt").open('w+') as f:
        json.dump(args.__dict__, f)

    package_status = get_package_status()

    if given_options:
        option_set = given_options
    else:
        option_set = [] 
        dataset_options = list(datasets)

        similarity_options = args.sim if type(args.sim) is list else [args.sim]
        fs_options = args.fsmethod if type(args.fsmethod) is list else [args.fsmethod]
        fingerprint_options = args.fingerprintmethod if type(args.fingerprintmethod) is list else [args.fingerprintmethod]
        fingerprint_bins_options = args.fingerprintbins if type(args.fingerprintbins) is list else [args.fingerprintbins]
        window_size_options = args.window_size if type(args.window_size) is list else [args.window_size]
        sim_gap_options = args.sim_gap if type(args.sim_gap) is list else [args.sim_gap]
        fp_gap_options = args.fp_gap if type(args.fp_gap) is list else [args.fp_gap]
        na_fp_gap_options = args.na_fp_gap if type(args.na_fp_gap) is list else [args.na_fp_gap]
        ob_gap_options = args.ob_gap if type(args.ob_gap) is list else [args.ob_gap]
        sim_stdevs_options = args.sim_stdevs if type(args.sim_stdevs) is list else [args.sim_stdevs]
        min_sim_stdev_options = args.min_sim_stdev if type(args.min_sim_stdev) is list else [args.min_sim_stdev]
        max_sim_stdev_options = args.max_sim_stdev if type(args.max_sim_stdev) is list else [args.max_sim_stdev]
        buffer_ratio_options = args.buffer_ratio if type(args.buffer_ratio) is list else [args.buffer_ratio]

        if not args.experimentopts:
            classifier_options = list(itertools.product(similarity_options,
                                                    fs_options,
                                                    fingerprint_options,
                                                    fingerprint_bins_options,
                                                    window_size_options,
                                                    sim_gap_options,
                                                    fp_gap_options,
                                                    na_fp_gap_options,
                                                    ob_gap_options,
                                                    sim_stdevs_options,
                                                    min_sim_stdev_options,
                                                    max_sim_stdev_options,
                                                    buffer_ratio_options
                                                    ))
        else:
            classifier_options = list(itertools.product(fingerprint_bins_options,
                                                    window_size_options,
                                                    sim_gap_options,
                                                    fp_gap_options,
                                                    na_fp_gap_options,
                                                    ob_gap_options,
                                                    sim_stdevs_options,
                                                    min_sim_stdev_options,
                                                    max_sim_stdev_options,
                                                    buffer_ratio_options
                                                    ))

        for ds_name, ds_type in dataset_options:
            # If we are reusing, find out what seeds are already in use
            # otherwise, make a new one.
            if args.seedaction == 'reuse':
                seed_location = raw_data_path / ds_type / ds_name / "seeds"
                if not seed_location.exists():
                    seeds = []
                else:
                    seeds = [int(str(f.stem)) for f in seed_location.iterdir() if f.is_dir()]
                for i in range(num_seeds):
                    if i < len(seeds):
                        continue
                    seeds.append(base_seeds[i])
                if len(seeds) < 1:
                    raise ValueError(f"Reuse seeds selected by no seeds exist for data set {ds_name}")
            
            for seed in seeds:
                if not args.experimentopts:
                    for sim_opt, fs_opt, fingerprint_opt, fingerprint_bins_opt, ws_opt, sim_gap_opt, fp_gap_opt, na_fp_gap_opt, ob_gap_opt, sim_std_opt, min_sim_opt, max_sim_opt, br_opt in classifier_options:
                        option = {
                            'base_output_path': base_output_path,
                            'raw_data_path': raw_data_path,
                            'data_name': ds_name,
                            'data_type': ds_type,
                            'max_rows': args.maxrows,
                            'seed': seed,
                            'seed_action': args.seedaction,
                            'package_status': package_status,
                            'log_name': log_name,
                            'pyinstrument': args.pyinstrument,
                            'experiment_name': args.experimentname,
                            'repeats': args.repeats,
                            'concept_max': args.concept_max,
                            'concept_length': args.concept_length,
                            'repeatproportion': args.repeatproportion,
                            'framework': "system",
                            'isources': args.isources,
                            'ifeatures': args.ifeatures,
                            'optdetect': args.optdetect,
                            'optselect': args.optselect,
                            'similarity_option': sim_opt,
                            'window_size': ws_opt,
                            'similarity_gap': sim_gap_opt,
                            'fp_gap': fp_gap_opt,
                            'nonactive_fp_gap': na_fp_gap_opt,
                            'observation_gap': ob_gap_opt,
                            'take_observations': ob_gap_opt != 0,
                            'similarity_stdev_thresh': sim_std_opt,
                            'similarity_stdev_min': min_sim_opt,
                            'similarity_stdev_max': max_sim_opt,
                            'buffer_ratio': br_opt,
                            'fs_method': fs_opt,
                            'fingerprint_method': fingerprint_opt,
                            'fingerprint_bins': fingerprint_bins_opt,
                        }
                        stream, stream_concepts, length, classes = make_stream(option)
                        option_set.append(option)
                else:
                    for fingerprint_bins_opt, ws_opt, sim_gap_opt, fp_gap_opt, na_fp_gap_opt, ob_gap_opt, sim_std_opt, min_sim_opt, max_sim_opt, br_opt in classifier_options:
                        for exp_fingerprint, exp_fsmethod, sim_opt in [('cache', 'default', 'metainfo'), ('cache', 'fisher', 'metainfo'), ('cache', 'CacheMIHy', 'metainfo'), ('cachehistogram', 'Cachehistogram_MI', 'metainfo'), ('cachesketch', 'sketch_MI', 'metainfo'), ('cachesketch', 'sketch_covMI', 'metainfo'), ('cachesketch', 'sketch_MI', 'sketch'), ('cachesketch', 'sketch_covMI', 'sketch')]:
                            # Only need to run default and fisher on one bin size, as it doesn't do anything
                            if exp_fsmethod in ['default', 'fisher'] and fingerprint_bins_opt != fingerprint_bins_options[0]:
                                continue
                            option = {
                                'base_output_path': base_output_path,
                                'raw_data_path': raw_data_path,
                                'data_name': ds_name,
                                'data_type': ds_type,
                                'max_rows': args.maxrows,
                                'seed': seed,
                                'seed_action': args.seedaction,
                                'package_status': package_status,
                                'log_name': log_name,
                                'pyinstrument': args.pyinstrument,
                                'experiment_name': args.experimentname,
                                'repeats': args.repeats,
                                'concept_max': args.concept_max,
                                'concept_length': args.concept_length,
                                'repeatproportion': args.repeatproportion,
                                'framework': "system",
                                'isources': args.isources,
                                'ifeatures': args.ifeatures,
                                'optdetect': args.optdetect,
                                'optselect': args.optselect,
                                'similarity_option': sim_opt,
                                'window_size': ws_opt,
                                'similarity_gap': sim_gap_opt,
                                'fp_gap': fp_gap_opt,
                                'nonactive_fp_gap': na_fp_gap_opt,
                                'observation_gap': ob_gap_opt,
                                'take_observations': ob_gap_opt != 0,
                                'similarity_stdev_thresh': sim_std_opt,
                                'similarity_stdev_min': min_sim_opt,
                                'similarity_stdev_max': max_sim_opt,
                                'buffer_ratio': br_opt,
                                'fs_method': exp_fsmethod,
                                'fingerprint_method': exp_fingerprint,
                                'fingerprint_bins': fingerprint_bins_opt,
                            }
                            stream, stream_concepts, length, classes = make_stream(option)
                            option_set.append(option)
    with (log_path / f"e{log_name}_option_set.txt").open('w+') as f:
        json.dump(option_set, f, cls=NpEncoder)
    if args.single:
        run_files = []
        for o in tqdm.tqdm(option_set, total=len(option_set), position=0, desc="Experiment", leave=True):
            run_files.append(process_option(o))
    else:
        tqdm.tqdm.set_lock(RLock())
        pool = mp.Pool(processes=args.cpu, initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),))
        # run_files = pool.map(process_option, option_set, chunksize=1)
        run_files = []
        pool_run = tqdm.tqdm(pool.imap(func=process_option, iterable=option_set, chunksize=1), total=len(option_set), position=0, desc="Experiment", leave=True)
        for result in pool_run:
            run_files.append(result)
            pool_run.refresh()
        print(run_files)
        pool.close()
    with (log_path / f"e{log_name}_run_files.txt").open('w+') as f:
        json.dump(run_files, f, cls=NpEncoder)

