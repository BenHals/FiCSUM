import warnings
from ConceptFingerprint.Classifier.simple_CD_classifier import CDClassifier
from ConceptFingerprint.Classifier.hoeffding_tree_shap import HoeffdingTreeSHAPClassifier
from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
from skmultiflow.data.sine_generator import SineGenerator
from ConceptFingerprint.utils import tree_ensemble_init, SingleTree
import shap
import numpy as np
import scipy


def test_simple_CDClassifier_same_res():
    # We know sometimes hoeffding tree raises numpy errors in
    # its naive bayes calculation.
    np.seterr(all='ignore')
    scipy.special.seterr(all='ignore')
    #Moneky patch shap to use scikit-multiflow hoeffding trees.
    # shap.explainers.tree.TreeEnsemble.__init__ =  tree_ensemble_init
    # shap.Tree = Tree
    shap.explainers._tree.TreeEnsemble.__init__ =  tree_ensemble_init
    shap.SingleTree = SingleTree
    hoeffding_tree = HoeffdingTreeClassifier()
    learner = HoeffdingTreeSHAPClassifier
    cd_classifier = CDClassifier(learner=learner, poisson=1)

    stream = SineGenerator(classification_function=0)
    X,y = stream.next_sample()
    cd_classifier.partial_fit(X, y, classes = [0, 1])
    hoeffding_tree.partial_fit(X, y, classes = [0, 1])

    # Test result is the same before any drift, then 
    # the same as a fresh classifier after drift.
    counter = 0
    for i in range(4000):
        if i % 1000 == 0:
            stream = SineGenerator(classification_function=counter)
            counter += 1
        X,y = stream.next_sample()

        with np.errstate(all='ignore'):
            p = hoeffding_tree.predict(X)
            assert cd_classifier.predict(X) == p
        cd_classifier.partial_fit(X, y, classes = [0, 1])
        with np.errstate(all='ignore'):
            hoeffding_tree.partial_fit(X, y, classes = [0, 1])
        if cd_classifier.found_change:
            hoeffding_tree = HoeffdingTreeClassifier()
            print("new")

# test_simple_CDClassifier_same_res()