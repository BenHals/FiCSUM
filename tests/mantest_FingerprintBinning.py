from ConceptFingerprint.Classifier.meta_info_classifier import (
    FingerprintBinning,
)
from ConceptFingerprint.Classifier.normalizer import Normalizer
import numpy as np

norm = Normalizer(fingerprint_constructor = FingerprintBinning)
stats = {
    "test_source1": {"test_feature1": np.random.normal(10, 1), "test_feature2": np.random.rand()},
    "test_source2": {"test_feature1": np.random.rand() * 10, "test_feature2": np.random.normal(0, 1)},
}
norm.add_stats(stats)
fp = FingerprintBinning(stats, norm)

for i in range(2, 100):
    stats = {
        "test_source1": {"test_feature1": np.random.normal(10, 1), "test_feature2": np.random.rand()},
        "test_source2": {"test_feature1": np.random.rand() * 10, "test_feature2": np.random.normal(0, 1)},
    }
    norm.add_stats(stats)
    fp.incorperate(stats)
for i in range(2, 100):
    stats = {
        "test_source1": {"test_feature1": np.random.normal(0, 1), "test_feature2": np.random.rand() * 10},
        "test_source2": {"test_feature1": np.random.rand() * 10, "test_feature2": np.random.normal(0, 1)},
    }
    norm.add_stats(stats)
    fp.incorperate(stats)

for s in ['test_source1', 'test_source2']:
    for f in ['test_feature1', 'test_feature2']:
        print(s, f, fp.fingerprint[s][f]['Range'], fp.fingerprint[s][f]['Bins'], fp.fingerprint[s][f]['Histogram'])