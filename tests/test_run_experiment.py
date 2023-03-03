from run_experiment import make_stream
import tqdm
import pathlib
import numpy as np
import pytest

@pytest.mark.skip(reason="Need Data")
def test_stream_similarity_using_seeds():
    option = {
        'base_output_path': pathlib.Path("\output"),
        'raw_data_path': pathlib.Path("\RawData"),
        'data_name': "cmc",
        'data_type': "Real",
        'seed': 5,
        "max_rows": -1,
        "repeats": 3,
        "concept_max": 6,
        "repeatproportion": 1
    }
    option_2 = {
        'base_output_path': pathlib.Path("\output"),
        'raw_data_path': pathlib.Path("\RawData"),
        'data_name': "cmc",
        'data_type': "Real",
        'seed': 1000,
        "max_rows": -1,
        "repeats": 3,
        "concept_max": 6,
        "repeatproportion": 1
    }

    # Stream and Stream_2 should be equal, Stream_3 should be different.
    stream, stream_concepts, length, classes = make_stream(option)
    stream_2, stream_concepts, length, classes = make_stream(option)
    stream_3, stream_concepts, length, classes = make_stream(option_2)

    same_equals = []
    different_equals = []
    for i in tqdm.tqdm(range(option['length'])):
        X, y = stream.next_sample()
        X_2, y_2 = stream_2.next_sample()
        X_3, y_3 = stream_3.next_sample()
        same_equals.append((X == X_2).all() and (y == y_2).all())
        different_equals.append((X == X_3).all() and (y == y_3).all())

    assert np.array(same_equals).all()
    assert not np.array(different_equals).all()

# @pytest.mark.skip(reason="Need to make directories")
def test_stream_similarity_using_seeds_synth():
    option = {
        'base_output_path': pathlib.Path("\output"),
        'raw_data_path': pathlib.Path("\RawData"),
        'data_name': "RTREESAMPLE",
        'data_type': "Synthetic",
        'seed': 5,
        "max_rows": -1,
        "repeats": 2,
        "concept_max": 6,
        "repeatproportion": 1,
        "concept_length": 1000,
        "shuffleconcepts": False,
        "drift_width": 0,
    }
    option_2 = {
        'base_output_path': pathlib.Path("\output"),
        'raw_data_path': pathlib.Path("\RawData"),
        'data_name': "RTREESAMPLE",
        'data_type': "Synthetic",
        'seed': 1000,
        "max_rows": -1,
        "repeats": 2,
        "concept_max": 6,
        "repeatproportion": 1,
        "concept_length": 1000,
        "shuffleconcepts": False,
        "drift_width": 0,
    }

    # Stream and Stream_2 should be equal, Stream_3 should be different.
    stream, stream_concepts, length, classes = make_stream(option)
    stream_2, stream_concepts, length, classes = make_stream(option)
    stream_3, stream_concepts, length, classes = make_stream(option_2)

    same_equals = []
    different_equals = []
    for i in tqdm.tqdm(range(option['length'])):
        X, y = stream.next_sample()
        X_2, y_2 = stream_2.next_sample()
        X_3, y_3 = stream_3.next_sample()
        same_equals.append((X == X_2).all() and (y == y_2).all())
        different_equals.append((X == X_3).all() and (y == y_3).all())

    assert np.array(same_equals).all()
    assert  not np.array(different_equals).all()

# test_stream_similarity_using_seeds()
# test_stream_similarity_using_seeds_synth()