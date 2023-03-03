import warnings
from ConceptFingerprint.Classifier.hoeffding_tree_shap import HoeffdingTreeSHAP, HoeffdingTreeSHAPClassifier
from ConceptFingerprint.utils import tree_ensemble_init, SingleTree
from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
from skmultiflow.data.sine_generator import SineGenerator
import shap
import numpy as np
import scipy


def test_back_compat_constructor():
    max_byte_size = 10000
    memory_estimate_period = 10000
    grace_period = 300
    split_criterion = "gini"
    split_confidence = 0.001
    tie_threshold = 0.001
    binary_split = True
    stop_mem_management = True
    remove_poor_atts = True
    no_preprune = True
    leaf_prediction = 'nb'
    nb_threshold = 1
    nominal_attributes = []
    #Moneky patch shap to use scikit-multiflow hoeffding trees.
    # shap.explainers.tree.TreeEnsemble.__init__ =  tree_ensemble_init
    # shap.Tree = Tree
    shap.explainers._tree.TreeEnsemble.__init__ =  tree_ensemble_init
    shap.SingleTree = SingleTree
    new_constructed = HoeffdingTreeSHAPClassifier(max_byte_size=max_byte_size,
                                   memory_estimate_period=memory_estimate_period,
                                   grace_period=grace_period,
                                   split_criterion=split_criterion,
                                   split_confidence=split_confidence,
                                   tie_threshold=tie_threshold,
                                   binary_split=binary_split,
                                   stop_mem_management=stop_mem_management,
                                   remove_poor_atts=remove_poor_atts,
                                   no_preprune=no_preprune,
                                   leaf_prediction=leaf_prediction,
                                   nb_threshold=nb_threshold,
                                   nominal_attributes=nominal_attributes)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        backcompat = HoeffdingTreeSHAP(max_byte_size=max_byte_size,
                                    memory_estimate_period=memory_estimate_period,
                                    grace_period=grace_period,
                                    split_criterion=split_criterion,
                                    split_confidence=split_confidence,
                                    tie_threshold=tie_threshold,
                                    binary_split=binary_split,
                                    stop_mem_management=stop_mem_management,
                                    remove_poor_atts=remove_poor_atts,
                                    no_preprune=no_preprune,
                                    leaf_prediction=leaf_prediction,
                                    nb_threshold=nb_threshold,
                                    nominal_attributes=nominal_attributes)
    assert new_constructed.max_byte_size == backcompat.max_byte_size
    assert new_constructed.memory_estimate_period == backcompat.memory_estimate_period
    assert new_constructed.grace_period == backcompat.grace_period
    assert new_constructed.split_criterion == backcompat.split_criterion
    assert new_constructed.split_confidence == backcompat.split_confidence
    assert new_constructed.tie_threshold == backcompat.tie_threshold
    assert new_constructed.binary_split == backcompat.binary_split
    assert new_constructed.stop_mem_management == backcompat.stop_mem_management
    assert new_constructed.remove_poor_atts == backcompat.remove_poor_atts
    assert new_constructed.no_preprune == backcompat.no_preprune
    assert new_constructed.leaf_prediction == backcompat.leaf_prediction
    assert new_constructed.nb_threshold == backcompat.nb_threshold
    assert new_constructed.nominal_attributes == backcompat.nominal_attributes

def test_same_result_as_original():
    #Moneky patch shap to use scikit-multiflow hoeffding trees.
    # shap.explainers.tree.TreeEnsemble.__init__ =  tree_ensemble_init
    # shap.Tree = Tree
    shap.explainers._tree.TreeEnsemble.__init__ =  tree_ensemble_init
    
    shap.SingleTree = SingleTree
    shap_tree = HoeffdingTreeSHAPClassifier()
    original_tree = HoeffdingTreeClassifier()
    stream = SineGenerator()
    X,y = stream.next_sample()
    shap_tree.partial_fit(X, y, classes = [0, 1])
    original_tree.partial_fit(X, y, classes = [0, 1])
    np.seterr(all='ignore')
    scipy.special.seterr(all='ignore')
    for i in range(1000):
        X,y = stream.next_sample()
        with np.errstate(all='ignore'):
            ot_p = original_tree.predict(X)
            st_p = shap_tree.predict(X)
            assert st_p == ot_p
            shap_tree.partial_fit(X, y)
            original_tree.partial_fit(X, y)
def test_shap_model():
    #Moneky patch shap to use scikit-multiflow hoeffding trees.
    # shap.explainers.tree.TreeEnsemble.__init__ =  tree_ensemble_init
    # shap.Tree = Tree
    shap.explainers._tree.TreeEnsemble.__init__ =  tree_ensemble_init
    shap.SingleTree = SingleTree
    shap_tree = HoeffdingTreeSHAPClassifier()
    stream = SineGenerator()
    
    X,y = stream.next_sample()
    with np.errstate(all='ignore'):
        shap_tree.partial_fit(X, y, classes = [0, 1])
    for i in range(1000):
        X,y = stream.next_sample()
        with np.errstate(all='ignore'):
            shap_tree.partial_fit(X, y)
    shap_model = shap_tree.shap_model

    assert len(shap_model.model.trees[0].children_left) == shap_tree._decision_node_cnt + shap_tree._active_leaf_node_cnt + shap_tree._inactive_leaf_node_cnt

    # print(shap_model.model.trees[0].children_left)
    # print(shap_model.model.trees[0].children_right)
    # print(shap_model.model.trees[0].values)
    # print(shap_model.model.trees[0].node_sample_weight)
    # print(shap_model.model.trees[0].thresholds) 

    # print(shap_tree.get_model_description())



# test_back_compat_constructor()
# test_same_result_as_original()
# test_shap_model()