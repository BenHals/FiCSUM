"""
Load a real world data set with distinguished concepts.
Arrage into a single stream with recurring concepts.
"""
import argparse
import pathlib
import pickle
import csv
import shutil
import math

import pandas as pd
import numpy as np
import json

from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.data_stream import DataStream
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.agrawal_generator import AGRAWALGenerator
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.base_stream import Stream
from skmultiflow.utils import check_random_state
from ConceptFingerprint.Exploration.random_tree_generator_sample_orig import RandomTreeGeneratorSample
from ConceptFingerprint.Exploration.featureweightExpGen import FeatureWeightExpGenerator
from ConceptFingerprint.Exploration.SigNoiseExpGen import SigNoiseGenerator
from ConceptFingerprint.Exploration.RTREE_HP_Features_generator import RandomTreeGeneratorHPFeatureSample
from ConceptFingerprint.Exploration.hyper_plane_generator_sample_v5 import HyperplaneSampleGenerator


class AbruptDriftStream(Stream):
    def __init__(self, streams, length, random_state = None, width = None):
        super(AbruptDriftStream, self).__init__()

        stream = streams[0][2]
        self.abrupt = True
        self.drift_width = 0
        if width is not None:
            self.abrupt = False
            self.drift_width = width
        self.n_samples = length
        self.n_targets = stream.n_targets
        self.n_features = stream.n_features
        self.n_num_features = stream.n_num_features
        self.n_cat_features = stream.n_cat_features
        self.n_classes = stream.n_classes
        self.cat_features_idx = stream.cat_features_idx
        self.feature_names = stream.feature_names
        self.target_names = stream.target_names
        self.target_values = stream.target_values
        self.n_targets = stream.n_targets
        self.target_values = set()
        for _, _, stream, _ in streams:
            try:
                stream_target_values = stream._get_target_values()
            except:
                stream_target_values = stream.target_values
            for stv in stream_target_values:
                self.target_values.add(stv)
        self.name = "AbruptStream"

        self.random_state = random_state
        self._random_state = None   # This is the actual random_state object used internally
        self.streams = streams
        self.length = length
        self.stream_idx = 0


        self._prepare_for_use()

    def _get_target_values(self):
        return self.target_values

    def _prepare_for_use(self):
        self._random_state = check_random_state(self.random_state)

    def prepare_for_use(self):
        self._prepare_for_use()

    def n_remaining_samples(self):
        """ Returns the estimated number of remaining samples.
        Returns
        -------
        int
            Remaining number of samples. -1 if infinite (e.g. generator)
        """
        return self.n_samples - self.sample_idx

    def has_more_samples(self):
        """ Checks if stream has more samples.
        Returns
        -------
        Boolean
            True if stream has more samples.
        """
        return self.n_remaining_samples() > 0

    def is_restartable(self):
        """ Determine if the stream is restartable.
         Returns
         -------
         Boolean
            True if stream is restartable.
         """
        return True

    def next_sample(self, batch_size=1):
        """ Returns next sample from the stream.
        Parameters
        ----------
        batch_size: int (optional, default=1)
            The number of samples to return.
        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix
            for the batch_size samples that were requested.
        """
        self.current_sample_x = np.zeros((batch_size, self.n_features))
        self.current_sample_y = np.zeros((batch_size, self.n_targets))

        for j in range(batch_size):
            have_correct_concept = False
            while not have_correct_concept:
                current_concept = self.streams[self.stream_idx]
                if self.sample_idx > current_concept[1]:
                    self.stream_idx += 1
                elif self.sample_idx < current_concept[0]:
                    self.stream_idx -= 1
                else:
                    have_correct_concept = True

            if not self.abrupt:
                if self.stream_idx < len(self.streams)-1 and self.sample_idx > (current_concept[1] - self.drift_width / 2):
                    num_until_end = current_concept[1] - self.sample_idx
                    concept_chance = ((self.drift_width/ 2) + num_until_end) / self.drift_width
                    rand = self._random_state.rand()
                    if rand <= concept_chance:
                        have_correct_concept = True
                    else:
                        self.stream_idx += 1
                        current_concept = self.streams[self.stream_idx]
                if self.stream_idx > 0 and self.sample_idx < (current_concept[0] + self.drift_width / 2):
                    num_after_start = self.sample_idx - current_concept[0]
                    concept_chance = ((self.drift_width/ 2) + num_after_start) / self.drift_width
                    rand = self._random_state.rand()
                    if rand <= concept_chance:
                        have_correct_concept = True
                    else:
                        self.stream_idx -= 1
                        current_concept = self.streams[self.stream_idx]
            
            if current_concept[2].n_remaining_samples() == 0:
                current_concept[2].restart()
            
            X,y = current_concept[2].next_sample()
                


            self.current_sample_x[j, :] = X
            self.current_sample_y[j, :] = y
            self.sample_idx += 1

        return self.current_sample_x, self.current_sample_y.flatten()

    def restart(self):
        self._random_state = check_random_state(self.random_state)
        self.sample_idx = 0
        self.stream_idx = 0
        for s in self.streams:
            s.restart()

def RTREEGenerator(classification_function, random_state):
    return RandomTreeGenerator(tree_random_state=classification_function, sample_random_state=random_state)
def STAGGERGeneratorWrapper(classification_function, random_state):
    return STAGGERGenerator(classification_function=classification_function%3, random_state=random_state)
def SEAGeneratorWrapper(classification_function, random_state):
    return SEAGenerator(classification_function=classification_function%4, random_state=random_state)
def AGRAWALGeneratorWrapper(classification_function, random_state):
    return AGRAWALGenerator(classification_function=classification_function%10, random_state=random_state)

def RTREESAMPLEGenerator(sampler_features, intra_concept_dist='dist', inter_concept_dist='uniform'):
    if sampler_features:
        return lambda classification_function, random_state: RandomTreeGeneratorSample(tree_random_state=classification_function, intra_concept_dist=intra_concept_dist, inter_concept_dist=inter_concept_dist, sampler_random_state = random_state, sampler_features = sampler_features, strength=1)
    return lambda classification_function, random_state: RandomTreeGeneratorSample(tree_random_state=classification_function, intra_concept_dist=intra_concept_dist, inter_concept_dist=inter_concept_dist, sampler_random_state = random_state, strength=1)
def FeatureWeightExpGen(sampler_features, intra_concept_dist='dist', inter_concept_dist='uniform'):
    if sampler_features:
        return lambda classification_function, random_state: FeatureWeightExpGenerator(tree_random_state=classification_function, intra_concept_dist=intra_concept_dist, inter_concept_dist=inter_concept_dist, sampler_random_state = random_state, sampler_features = sampler_features, strength=1)
    return lambda classification_function, random_state: FeatureWeightExpGenerator(tree_random_state=classification_function, intra_concept_dist=intra_concept_dist, inter_concept_dist=inter_concept_dist, sampler_random_state = random_state, strength=1)
def SigNoiseGen(signal_noise_ratio):
    return lambda classification_function, random_state: SigNoiseGenerator(tree_random_state=classification_function, sampler_random_state = random_state, signal_noise_ratio=signal_noise_ratio)
def RTREESAMPLEHPGenerator(sampler_features, intra_concept_dist='dist', inter_concept_dist='uniform'):
    if sampler_features:
        return lambda classification_function, random_state: RandomTreeGeneratorHPFeatureSample(tree_random_state=classification_function, intra_concept_dist=intra_concept_dist, inter_concept_dist=inter_concept_dist, sampler_random_state = random_state, sampler_features = sampler_features, strength=1)
    return lambda classification_function, random_state: RandomTreeGeneratorHPFeatureSample(tree_random_state=classification_function, intra_concept_dist=intra_concept_dist, inter_concept_dist=inter_concept_dist, sampler_random_state = random_state, strength=1)
def HPLANESAMPLEGenerator(sampler_features):
    if sampler_features:
        return lambda classification_function, random_state: HyperplaneSampleGenerator(random_state=classification_function, n_features=8, n_drift_features=0, mag_change=0, sampler_random_state = random_state, sampler_features = sampler_features)
    return lambda classification_function, random_state: HyperplaneSampleGenerator(random_state=classification_function, n_features=8, n_drift_features=0, mag_change=0, sampler_random_state = random_state)

def HPLANEGenerator(classification_function, random_state):
    return HyperplaneGenerator(random_state=classification_function, n_features=8, n_drift_features=0, mag_change=0)
def RBFGenerator(classification_function, random_state):
    return RandomRBFGenerator(model_random_state=classification_function, sample_random_state=random_state)
def RBFGeneratorDifficulty(difficulty):
    n_centroids = difficulty * 5 + 15
    return lambda classification_function, random_state: RandomRBFGenerator(model_random_state=classification_function, sample_random_state=random_state, n_centroids=n_centroids, n_classes=4)
def RTREEGeneratorDifficulty(difficulty = 0):
    return lambda classification_function, random_state: RandomTreeGenerator(tree_random_state=classification_function, sample_random_state=random_state, max_tree_depth=difficulty+1, min_leaf_depth=difficulty)
def RTREESAMPLEGeneratorDifficulty(sampler_features, difficulty = 0, strength = 1):
    if sampler_features:
        return lambda classification_function, random_state: RandomTreeGeneratorSample(tree_random_state=classification_function, sampler_random_state = random_state, sampler_features = sampler_features, max_tree_depth=difficulty+2, min_leaf_depth=difficulty, strength = strength)
    return lambda classification_function, random_state: RandomTreeGeneratorSample(tree_random_state=classification_function, sampler_random_state = random_state, max_tree_depth=difficulty+2, min_leaf_depth=difficulty, strength = strength)
    
    

def create_synthetic_concepts(path, name, seed):
    stream_generator = None
    num_concepts = None
    sampler_features = None
    if name == "STAGGER":
        stream_generator = STAGGERGeneratorWrapper
        num_concepts = 10
    if name == "STAGGERS":
        stream_generator = STAGGERGeneratorWrapper
        num_concepts = 3
    if name == "ARGWAL":
        stream_generator = AGRAWALGeneratorWrapper
        num_concepts = 10
    if name == "SEA":
        stream_generator = SEAGeneratorWrapper
        num_concepts = 10
    if name == "RTREE":
        stream_generator = RTREEGenerator
        num_concepts = 10
    if name == "RTREEEasy":
        stream_generator = RTREEGeneratorDifficulty(difficulty=0)
        num_concepts = 10
    if name == "RTREEEasySAMPLE":
        stream_generator = RTREESAMPLEGeneratorDifficulty(sampler_features, difficulty=0, strength=0.1)
        num_concepts = 10
    if name == "RTREEMedSAMPLE":
        stream_generator = RTREESAMPLEGeneratorDifficulty(sampler_features, difficulty=0, strength=0.2)
        num_concepts = 10
    if name == "SynEasyF":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['frequency'], difficulty=0)
        num_concepts = 10
    if name == "SynEasyA":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['autocorrelation'], difficulty=0)
        num_concepts = 10
    if name == "SynEasyD":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['distribution'], difficulty=0)
        num_concepts = 10
    if name == "SynEasyAF":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['autocorrelation', 'frequency'], difficulty=0)
        num_concepts = 10
    if name == "SynEasyDA":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['distribution', 'autocorrelation'], difficulty=0)
        num_concepts = 10
    if name == "SynEasyDF":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['distribution', 'frequency'], difficulty=0)
        num_concepts = 10
    if name == "SynEasyDAF":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['distribution', 'autocorrelation', 'frequency'], difficulty=0)
        num_concepts = 10
    if name == "RTREESAMPLE":
        stream_generator = RTREESAMPLEGenerator(sampler_features)
        num_concepts = 10
    if name == "HPLANE":
        stream_generator = HPLANEGenerator
        num_concepts = 10
    if name == "RBF":
        stream_generator = RBFGenerator
        num_concepts = 10
    if name == "RBFEasy":
        stream_generator = RBFGeneratorDifficulty(difficulty=0)
        num_concepts = 10
    if name == "RBFMed":
        stream_generator = RBFGeneratorDifficulty(difficulty=2)
        num_concepts = 10
    if name == "HPLANESAMPLE":
        stream_generator = HPLANESAMPLEGenerator(sampler_features)
        num_concepts = 10
    if name == "RTREESAMPLE-UU":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='uniform', inter_concept_dist='uniform')
        num_concepts = 10
    if name == "RTREESAMPLE-UN":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='uniform', inter_concept_dist='norm')
        num_concepts = 10
    if name == "RTREESAMPLE-UD":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='uniform', inter_concept_dist='dist')
        num_concepts = 10
    if name == "RTREESAMPLE-NU":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='norm', inter_concept_dist='uniform')
        num_concepts = 10
    if name == "RTREESAMPLE-NN":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='norm', inter_concept_dist='norm')
        num_concepts = 10
    if name == "RTREESAMPLE-ND":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='norm', inter_concept_dist='dist')
        num_concepts = 10
    if name == "RTREESAMPLE-DU":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='dist', inter_concept_dist='uniform')
        num_concepts = 10
    if name == "RTREESAMPLE-DN":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='dist', inter_concept_dist='norm')
        num_concepts = 10
    if name == "RTREESAMPLE-DD":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='dist', inter_concept_dist='dist')
        num_concepts = 10
    if name == "RTREESAMPLE-UB":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='uniform', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "RTREESAMPLE-NB":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='norm', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "RTREESAMPLE-DB":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='dist', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "RTREESAMPLEHP-A":
        stream_generator = RTREESAMPLEHPGenerator(sampler_features=['f1', 'f2', 'f3', 'f4'], intra_concept_dist='dist', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "RTREESAMPLEHP-23":
        stream_generator = RTREESAMPLEHPGenerator(sampler_features=['f2', 'f3'], intra_concept_dist='dist', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "RTREESAMPLEHP-14":
        stream_generator = RTREESAMPLEHPGenerator(sampler_features=['f1', 'f4'], intra_concept_dist='dist', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "FeatureWeightExpGenerator":
        stream_generator = FeatureWeightExpGen(sampler_features, intra_concept_dist='dist', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "SigNoiseGenerator-1":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.1)
        num_concepts = 10
    if name == "SigNoiseGenerator-2":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.2)
        num_concepts = 10
    if name == "SigNoiseGenerator-3":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.3)
        num_concepts = 10
    if name == "SigNoiseGenerator-4":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.4)
        num_concepts = 10
    if name == "SigNoiseGenerator-5":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.5)
        num_concepts = 10
    if name == "SigNoiseGenerator-6":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.6)
        num_concepts = 10
    if name == "SigNoiseGenerator-7":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.7)
        num_concepts = 10
    if name == "SigNoiseGenerator-8":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.8)
        num_concepts = 10
    if name == "SigNoiseGenerator-9":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.9)
        num_concepts = 10
    if name == "SigNoiseGenerator-10":
        stream_generator = SigNoiseGen(signal_noise_ratio=1.0)
        num_concepts = 10
    if stream_generator is None:
        raise ValueError("name not valid for a dataset")
    print(num_concepts)
    if seed is None:
        seed = np.random.randint(0, 1000)
    for c in range(num_concepts):
        concept = stream_generator(classification_function = seed + c, random_state=seed+c)
        with (path / f"concept_{c}.pickle").open("wb") as f:
            pickle.dump(concept, f)
        
        if hasattr(concept, 'get_data'):
            with (path / f"data_{c}.json").open("w") as f:
                json.dump(concept.get_data(), f)



def load_synthetic_concepts(name, seed, raw_data_path = None):
    data_path =  pathlib.Path(raw_data_path).resolve()

    file_path = data_path / name / "seeds" / str(seed)
    if not file_path.exists():
        file_path.mkdir(parents=True, exist_ok=True)
        create_synthetic_concepts(file_path, name, seed)

    concept_paths = list(file_path.glob('*concept*'))
    if len(concept_paths) == 0:
        create_synthetic_concepts(file_path, name, seed)
    concept_paths = list(file_path.glob('*concept*'))

    concepts = []
    for cp in concept_paths:
        with cp.open("rb") as f:
            concepts.append((pickle.load(f), cp.stem))
    concepts = sorted(concepts, key=lambda x: x[1])
    return concepts

def create_real_concept(csv_name, name, seed, nrows = -1, sort_examples = False):
    if nrows > 0:
        df = pd.read_csv(csv_name, nrows=nrows)
    else:
        df = pd.read_csv(csv_name)

    for c in df.columns:
        is_num = pd.api.types.is_numeric_dtype(df[c])
        if not is_num or c == df.columns[-1]:
            codes, unique = pd.factorize(df[c])
            df[c] = codes
    
    if sort_examples:
        df = df[df.columns[1:]]

    stream = DataStream(df)
    if not name.parent.exists():
        name.parent.mkdir(parents=True, exist_ok=True)
    with name.open("wb") as f:
        pickle.dump(stream, f)
    return stream



def load_real_concepts(name, seed, raw_data_path = None, nrows = -1, sort_examples = False):
    data_path =  pathlib.Path(raw_data_path)

    file_path = data_path / name

    concept_csv_paths = list(file_path.glob('*.csv'))
    for csv in concept_csv_paths:
        concept_name = csv.parent / "seeds" / str(seed) / f"concept_{csv.stem}.pickle"
        if not concept_name.exists():
            create_real_concept(csv, concept_name, seed, nrows, sort_examples=sort_examples)
    
    seed_path = file_path / "seeds" / str(seed)
    concept_paths = list(seed_path.glob('*concept*'))
    concepts = []
    for cp in concept_paths:
        if '_classes' in str(cp):
            continue
        with cp.open("rb") as f:
            concepts.append((pickle.load(f), cp.stem))
    concepts = sorted(concepts, key=lambda x: x[1])
    return concepts

def get_inorder_concept_ranges(concepts, seed, concept_length = 5000, repeats = 1, concept_max = -1, repeat_proportion = None, shuffle=True):
    idx = 0
    positions = []
    if shuffle:
        shuffle_random_state = np.random.RandomState(seed)
        shuffle_random_state.shuffle(concepts)
    for r in range(repeats):
        for i, (c,n) in enumerate(concepts):
            if concept_max > 0 and i >= concept_max:
                continue
            c_length = c.n_remaining_samples() if c.n_remaining_samples() != -1 else concept_length
            if repeat_proportion is not None:
                if repeat_proportion == -1:
                    c_length = c_length / repeats
                else:
                    c_length = c_length * repeat_proportion
            c_length = math.floor(c_length)
            start_idx = idx
            end_idx = start_idx + (c_length - 1)
            positions.append((start_idx, end_idx, c, n))
            idx = end_idx + 1
    
    return positions, idx












def stitch_concepts_inorder(concepts, concept_length = 10000):
    stream_A, n = concepts[0]
    # stream_A.prepare_for_use()
    stream_A.name = "A"
    stream_A_length = stream_A.n_remaining_samples() - 2 if stream_A.n_remaining_samples() != -1 else concept_length
    # stream_A_length = concept_length
    current_length = stream_A_length
    # print(stream_A_length)
    names = [(0, n)]
    
    for i in range(1, len(concepts)):
        stream_B, n = concepts[i]
        # stream_B.prepare_for_use()
        stream_B.name = "B"
        stream_B_length = stream_B.n_remaining_samples() - 2 if stream_B.n_remaining_samples() != -1 else concept_length
        # stream_B_length = concept_length
        new_stream = ConceptDriftStream(stream = stream_A, drift_stream= stream_B, position=current_length, width=1)
        stream_A = new_stream
        names.append((current_length, n))
        current_length += stream_B_length
        
    # print(current_length)
    return stream_A, names, current_length


def sep_csv(name, raw_data_path = None, concept_col = 'author'):
    data_path =  pathlib.Path(raw_data_path) if raw_data_path is not None else (pathlib.Path.cwd() / __file__).parent.parent / "RawData" / "Real"

    file_path = data_path / name
    print(file_path)

    concept_csv_paths = list(file_path.glob('*.csv'))[0]
    df = pd.read_csv(concept_csv_paths)
    concepts = df[concept_col]
    concepts = concepts.unique()
    for c in concepts:
        filtered = df[df[concept_col] == c]
        filtered = filtered.drop(concept_col, axis = 1)
        shuffled = filtered.sample(frac=1)
        shuffled.to_csv(f"{concept_csv_paths.parent / concept_csv_paths.stem}_{c}.csv")

def load_real_datastream(name, stream_type, seed, raw_data_path = None):
    # print(__file__)
    data_path =  pathlib.Path(raw_data_path) if raw_data_path is not None else (pathlib.Path.cwd() / __file__).parent.parent / "RawData" / "Real"

    file_path = data_path / name
    print(file_path)

    concept_csv_paths = list(file_path.glob('*.csv'))
    for csv in concept_csv_paths:
        # print(csv)
        print(csv.stem)
        concept_name = csv.parent / f"concept_{csv.stem}.pickle"
        stream = create_real_concept(csv, concept_name)
    return stream
    

if __name__ == "__main__":
    # concepts = load_synthetic_concepts("test", "STAGGER", 0)
    # print(concepts)
    # concepts = load_real_concepts("UCI-Wine", "REAL", 0)
    # positions, length = get_inorder_concept_ranges(concepts)
    # s = AbruptDriftStream(positions, length)
    # i = 0
    # while s.has_more_samples():
    #     X,y = s.next_sample()
    #     print(i)
    #     i += 1
    # print(concepts[0][0].n_remaining_samples())
    # stitch_concepts_inorder(concepts)
    sep_csv('qg')