import numpy as np
import numpy.core.multiarray
import scipy.special
import multiprocessing
import sys
import json
import os
import struct
import itertools
import warnings

# from shap.common import assert_import, record_import_error, DenseData, safe_isinstance, SHAPError
from shap.utils import assert_import, record_import_error, safe_isinstance
from shap import _cext

output_transform_codes = {
    "identity": 0,
    "logistic": 1,
    "logistic_nlogloss": 2,
    "squared_loss": 3
}

feature_perturbation_codes = {
    "interventional": 0,
    "tree_path_dependent": 1,
    "global_path_dependent": 2
}
def shap_values(self, X, y=None, tree_limit=None, approximate=False, check_additivity=True, from_call=False):
    """ Estimate the SHAP values for a set of samples.

    Parameters
    ----------
    X : numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
        A matrix of samples (# samples x # features) on which to explain the model's output.

    y : numpy.array
        An array of label values for each sample. Used when explaining loss functions.

    tree_limit : None (default) or int
        Limit the number of trees used by the model. By default None means no use the limit of the
        original model, and -1 means no limit.

    approximate : bool
        Run fast, but only roughly approximate the Tree SHAP values. This runs a method
        previously proposed by Saabas which only considers a single feature ordering. Take care
        since this does not have the consistency guarantees of Shapley values and places too
        much weight on lower splits in the tree.

    check_additivity : bool
        Run a validation check that the sum of the SHAP values equals the output of the model. This
        check takes only a small amount of time, and will catch potential unforeseen errors.
        Note that this check only runs right now when explaining the margin of the model.

    Returns
    -------
    array or list
        For models with a single output this returns a matrix of SHAP values
        (# samples x # features). Each row sums to the difference between the model output for that
        sample and the expected value of the model output (which is stored in the expected_value
        attribute of the explainer when it is constant). For models with vector outputs this returns
        a list of such matrices, one for each output.
    """
    if check_additivity and self.model.model_type == "pyspark":
        warnings.warn("check_additivity requires us to run predictions which is not supported with spark, ignoring." 
                        " Set check_additivity=False to remove this warning")
        check_additivity = False

    # see if we have a default tree_limit in place.
    if tree_limit is None:
        tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit

    # shortcut using the C++ version of Tree SHAP in XGBoost, LightGBM, and CatBoost
    if self.feature_perturbation == "tree_path_dependent" and self.model.model_type != "internal" and self.data is None:
        model_output_vals = None
        phi = None
        if self.model.model_type == "xgboost":
            import xgboost
            if not isinstance(X, xgboost.core.DMatrix):
                X = xgboost.DMatrix(X)
            if tree_limit == -1:
                tree_limit = 0
            try:
                phi = self.model.original_model.predict(
                    X, ntree_limit=tree_limit, pred_contribs=True,
                    approx_contribs=approximate, validate_features=False
                )
            except ValueError as e:
                    raise ValueError("This reshape error is often caused by passing a bad data matrix to SHAP. " \
                                        "See https://github.com/slundberg/shap/issues/580") from e

            if check_additivity and self.model.model_output == "raw":
                model_output_vals = self.model.original_model.predict(
                    X, ntree_limit=tree_limit, output_margin=True,
                    validate_features=False
                )

        elif self.model.model_type == "lightgbm":
            assert not approximate, "approximate=True is not supported for LightGBM models!"
            phi = self.model.original_model.predict(X, num_iteration=tree_limit, pred_contrib=True)
            # Note: the data must be joined on the last axis
            if self.model.original_model.params['objective'] == 'binary':
                if not from_call:
                    warnings.warn('LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray')
                phi = np.concatenate((0-phi, phi), axis=-1)
            if phi.shape[1] != X.shape[1] + 1:
                try:
                    phi = phi.reshape(X.shape[0], phi.shape[1]//(X.shape[1]+1), X.shape[1]+1)
                except ValueError as e:
                    raise Exception("This reshape error is often caused by passing a bad data matrix to SHAP. " \
                                        "See https://github.com/slundberg/shap/issues/580") from e

        elif self.model.model_type == "catboost": # thanks to the CatBoost team for implementing this...
            assert not approximate, "approximate=True is not supported for CatBoost models!"
            assert tree_limit == -1, "tree_limit is not yet supported for CatBoost models!"
            import catboost
            if type(X) != catboost.Pool:
                X = catboost.Pool(X, cat_features=self.model.cat_feature_indices)
            phi = self.model.original_model.get_feature_importance(data=X, fstr_type='ShapValues')

        # note we pull off the last column and keep it as our expected_value
        if phi is not None:
            if len(phi.shape) == 3:
                self.expected_value = [phi[0, i, -1] for i in range(phi.shape[1])]
                out = [phi[:, i, :-1] for i in range(phi.shape[1])]
            else:
                self.expected_value = phi[0, -1]
                out = phi[:, :-1]

            if check_additivity and model_output_vals is not None:
                self.assert_additivity(out, model_output_vals)

            return out

    # convert dataframes
    if safe_isinstance(X, "pandas.core.series.Series"):
        X = X.values
    elif safe_isinstance(X, "pandas.core.frame.DataFrame"):
        X = X.values
    flat_output = False
    if len(X.shape) == 1:
        flat_output = True
        X = X.reshape(1, X.shape[0])
    if X.dtype != self.model.input_dtype:
        X = X.astype(self.model.input_dtype)
    X_missing = np.isnan(X, dtype=np.bool_)
    assert isinstance(X, np.ndarray), "Unknown instance type: " + str(type(X))
    assert len(X.shape) == 2, "Passed input data matrix X must have 1 or 2 dimensions!"

    if tree_limit < 0 or tree_limit > self.model.values.shape[0]:
        tree_limit = self.model.values.shape[0]

    if self.model.model_output == "log_loss":
        assert y is not None, "Both samples and labels must be provided when model_output = \"log_loss\" (i.e. `explainer.shap_values(X, y)`)!"
        assert X.shape[0] == len(y), "The number of labels (%d) does not match the number of samples to explain (%d)!" % (len(y), X.shape[0])
    transform = self.model.get_transform()

    if self.feature_perturbation == "tree_path_dependent":
        assert self.model.fully_defined_weighting, "The background dataset you provided does not cover all the leaves in the model, " \
                                                    "so TreeExplainer cannot run with the feature_perturbation=\"tree_path_dependent\" option! " \
                                                    "Try providing a larger background dataset, or using feature_perturbation=\"interventional\"."

    # run the core algorithm using the C extension
    assert_import("cext")
    phi = np.zeros((X.shape[0], X.shape[1]+1, self.model.num_outputs))
    if not approximate:
        _cext.dense_tree_shap(
            self.model.children_left, self.model.children_right, self.model.children_default,
            self.model.features, self.model.thresholds, self.model.values, self.model.node_sample_weight,
            self.model.max_depth, X, X_missing, y, self.data, self.data_missing, tree_limit,
            self.model.base_offset, phi, feature_perturbation_codes[self.feature_perturbation],
            output_transform_codes[transform], False
        )
    else:
        _cext.dense_tree_saabas(
            self.model.children_left, self.model.children_right, self.model.children_default,
            self.model.features, self.model.thresholds, self.model.values,
            self.model.max_depth, tree_limit, self.model.base_offset, output_transform_codes[transform],
            X, X_missing, y, phi
        )

    # note we pull off the last column and keep it as our expected_value
    if self.model.num_outputs == 1:
        if self.expected_value is None and self.model.model_output != "log_loss":
            self.expected_value = phi[0, -1, 0]
        if flat_output:
            out = phi[0, :-1, 0]
        else:
            out = phi[:, :-1, 0]
    else:
        if self.expected_value is None and self.model.model_output != "log_loss":
            self.expected_value = [phi[0, -1, i] for i in range(phi.shape[2])]
        if flat_output:
            out = [phi[0, :-1, i] for i in range(self.model.num_outputs)]
        else:
            out = [phi[:, :-1, i] for i in range(self.model.num_outputs)]

    if check_additivity and self.model.model_output == "raw":
        self.assert_additivity(out, self.model.predict(X))

    # if our output format requires binary classificaiton to be represented as two outputs then we do that here
    if self.model.model_output == "probability_doubled":
        out = [-out, out]

    return out
def tree_ensemble_init(self, model, data=None, data_missing=None, model_output=None):
        self.model_type = "internal"
        self.trees = None
        less_than_or_equal = True
        self.base_offset = 0
        self.model_output = model_output
        self.objective = None # what we explain when explaining the loss of the model
        self.tree_output = None # what are the units of the values in the leaves of the trees
        self.internal_dtype = np.float64
        self.input_dtype = np.float64 # for sklearn we need to use np.float32 to always get exact matches to their predictions
        self.data = data
        self.data_missing = data_missing
        self.fully_defined_weighting = True # does the background dataset land in every leaf (making it valid for the tree_path_dependent method)
        self.tree_limit = None # used for limiting the number of trees we use by default (like from early stopping)
        self.num_stacked_models = 1 # If this is greater than 1 it means we have multiple stacked models with the same number of trees in each model (XGBoost multi-output style)
        self.cat_feature_indices = None # If this is set it tells us which features are treated categorically

        # we use names like keras
        objective_name_map = {
            "mse": "squared_error",
            "variance": "squared_error",
            "friedman_mse": "squared_error",
            "reg:linear": "squared_error",
            "reg:squarederror": "squared_error",
            "regression": "squared_error",
            "regression_l2": "squared_error",
            "mae": "absolute_error",
            "gini": "binary_crossentropy",
            "entropy": "binary_crossentropy",
            "reg:logistic": "binary_crossentropy",
            "binary:logistic": "binary_crossentropy",
            "binary_logloss": "binary_crossentropy",
            "binary": "binary_crossentropy"
        }

        tree_output_name_map = {
            "regression": "raw_value",
            "regression_l2": "squared_error",
            "reg:linear": "raw_value",
            "reg:squarederror": "raw_value",
            "reg:logistic": "log_odds",
            "binary:logistic": "log_odds",
            "binary_logloss": "log_odds",
            "binary": "log_odds"
        }

        if type(model) is dict and "trees" in model:
            # This allows a dictionary to be passed that represents the model.
            # this dictionary has several numerica paramters and also a list of trees
            # where each tree is a dictionary describing that tree
            if "internal_dtype" in model:
                self.internal_dtype = model["internal_dtype"]
            if "input_dtype" in model:
                self.input_dtype = model["input_dtype"]
            if "objective" in model:
                self.objective = model["objective"]
            if "tree_output" in model:
                self.tree_output = model["tree_output"]
            if "base_offset" in model:
                self.base_offset = model["base_offset"]
            self.trees = [SingleTree(t, data=data, data_missing=data_missing) for t in model["trees"]]
        elif type(model) is list and type(model[0]) == SingleTree: # old-style direct-load format
            self.trees = model
        elif safe_isinstance(model, ["sklearn.ensemble.RandomForestRegressor", "sklearn.ensemble.forest.RandomForestRegressor"]):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.ensemble.IsolationForest", "sklearn.ensemble.iforest.IsolationForest"]):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [IsoTree(e.tree_, f, scaling=scaling, data=data, data_missing=data_missing) for e, f in zip(model.estimators_, model.estimators_features_)]
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["pyod.models.iforest.IForest"]):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [IsoTree(e.tree_, f, scaling=scaling, data=data, data_missing=data_missing) for e, f in zip(model.detector_.estimators_, model.detector_.estimators_features_)]
            self.tree_output = "raw_value"
        elif safe_isinstance(model, "skopt.learning.forest.RandomForestRegressor"):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.ensemble.ExtraTreesRegressor", "sklearn.ensemble.forest.ExtraTreesRegressor"]):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, "skopt.learning.forest.ExtraTreesRegressor"):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.tree.DecisionTreeRegressor", "sklearn.tree.tree.DecisionTreeRegressor"]):
            self.internal_dtype = model.tree_.value.dtype.type
            self.input_dtype = np.float32
            self.trees = [SingleTree(model.tree_, data=data, data_missing=data_missing)]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.tree.DecisionTreeClassifier", "sklearn.tree.tree.DecisionTreeClassifier"]):
            self.internal_dtype = model.tree_.value.dtype.type
            self.input_dtype = np.float32
            self.trees = [SingleTree(model.tree_, normalize=True, data=data, data_missing=data_missing)]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif safe_isinstance(model, ["sklearn.ensemble.RandomForestClassifier", "sklearn.ensemble.forest.RandomForestClassifier"]):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif safe_isinstance(model, ["sklearn.ensemble.ExtraTreesClassifier", "sklearn.ensemble.forest.ExtraTreesClassifier"]): # TODO: add unit test for this case
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif safe_isinstance(model, ["sklearn.ensemble.GradientBoostingRegressor", "sklearn.ensemble.gradient_boosting.GradientBoostingRegressor"]):
            self.input_dtype = np.float32

            # currently we only support the mean and quantile estimators
            if safe_isinstance(model.init_, ["sklearn.ensemble.MeanEstimator", "sklearn.ensemble.gradient_boosting.MeanEstimator"]):
                self.base_offset = model.init_.mean
            elif safe_isinstance(model.init_, ["sklearn.ensemble.QuantileEstimator", "sklearn.ensemble.gradient_boosting.QuantileEstimator"]):
                self.base_offset = model.init_.quantile
            elif safe_isinstance(model.init_, "sklearn.dummy.DummyRegressor"):
                self.base_offset = model.init_.constant_[0]
            else:
                assert False, "Unsupported init model type: " + str(type(model.init_))

            self.trees = [SingleTree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing) for e in model.estimators_[:,0]]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.ensemble.HistGradientBoostingRegressor"]):
            import sklearn
            if self.model_output == "predict":
                self.model_output = "raw"
            self.input_dtype = sklearn.ensemble._hist_gradient_boosting.common.X_DTYPE
            self.base_offset = model._baseline_prediction
            self.trees = []
            for p in model._predictors:
                nodes = p[0].nodes
                # each node has values: ('value', 'count', 'feature_idx', 'threshold', 'missing_go_to_left', 'left', 'right', 'gain', 'depth', 'is_leaf', 'bin_threshold')
                tree = {
                    "children_left": np.array([-1 if n[9] else n[5] for n in nodes]),
                    "children_right": np.array([-1 if n[9] else n[6] for n in nodes]),
                    "children_default": np.array([-1 if n[9] else (n[5] if n[4] else n[6]) for n in nodes]),
                    "features": np.array([-2 if n[9] else n[2] for n in nodes]),
                    "thresholds": np.array([n[3] for n in nodes], dtype=np.float64),
                    "values": np.array([[n[0]] for n in nodes], dtype=np.float64),
                    "node_sample_weight": np.array([n[1] for n in nodes], dtype=np.float64),
                }
                self.trees.append(SingleTree(tree, data=data, data_missing=data_missing))
            self.objective = objective_name_map.get(model.loss, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.ensemble.HistGradientBoostingClassifier"]):
            import sklearn
            self.base_offset = model._baseline_prediction
            if hasattr(self.base_offset, "__len__") and self.model_output != "raw":
                raise Exception("Multi-output HistGradientBoostingClassifier models are not yet supported unless model_output=\"raw\". See GitHub issue #1028")
            self.input_dtype = sklearn.ensemble._hist_gradient_boosting.common.X_DTYPE
            self.num_stacked_models = len(model._predictors[0])
            if self.model_output == "predict_proba":
                if self.num_stacked_models == 1:
                    self.model_output = "probability_doubled" # with predict_proba we need to double the outputs to match
                else:
                    self.model_output = "probability"
            output_trees = [[] for i in range(self.num_stacked_models)]
            for p in model._predictors:
                for i in range(self.num_stacked_models):
                    nodes = p[i].nodes
                    # each node has values: ('value', 'count', 'feature_idx', 'threshold', 'missing_go_to_left', 'left', 'right', 'gain', 'depth', 'is_leaf', 'bin_threshold')
                    tree = {
                        "children_left": np.array([-1 if n[9] else n[5] for n in nodes]),
                        "children_right": np.array([-1 if n[9] else n[6] for n in nodes]),
                        "children_default": np.array([-1 if n[9] else (n[5] if n[4] else n[6]) for n in nodes]),
                        "features": np.array([-2 if n[9] else n[2] for n in nodes]),
                        "thresholds": np.array([n[3] for n in nodes], dtype=np.float64),
                        "values": np.array([[n[0]] for n in nodes], dtype=np.float64),
                        "node_sample_weight": np.array([n[1] for n in nodes], dtype=np.float64),
                    }
                    output_trees[i].append(SingleTree(tree, data=data, data_missing=data_missing))
            self.trees = list(itertools.chain.from_iterable(output_trees))
            self.objective = objective_name_map.get(model.loss, None)
            self.tree_output = "log_odds"
        elif safe_isinstance(model, ["sklearn.ensemble.GradientBoostingClassifier","sklearn.ensemble._gb.GradientBoostingClassifier", "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier"]):
            self.input_dtype = np.float32

            # TODO: deal with estimators for each class
            if model.estimators_.shape[1] > 1:
                assert False, "GradientBoostingClassifier is only supported for binary classification right now!"

            # currently we only support the logs odds estimator
            if safe_isinstance(model.init_, ["sklearn.ensemble.LogOddsEstimator", "sklearn.ensemble.gradient_boosting.LogOddsEstimator"]):
                self.base_offset = model.init_.prior
                self.tree_output = "log_odds"
            elif safe_isinstance(model.init_, "sklearn.dummy.DummyClassifier"):
                self.base_offset = scipy.special.logit(model.init_.class_prior_[1]) # with two classes the trees only model the second class. # pylint: disable=no-member
                self.tree_output = "log_odds"
            else:
                assert False, "Unsupported init model type: " + str(type(model.init_))

            self.trees = [SingleTree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing) for e in model.estimators_[:,0]]
            self.objective = objective_name_map.get(model.criterion, None)
        elif "pyspark.ml" in str(type(model)):
            assert_import("pyspark")
            self.model_type = "pyspark"
            # model._java_obj.getImpurity() can be gini, entropy or variance.
            self.objective = objective_name_map.get(model._java_obj.getImpurity(), None)
            if "Classification" in str(type(model)):
                normalize = True
                self.tree_output = "probability"
            else:
                normalize = False
                self.tree_output = "raw_value"
            # Spark Random forest, create 1 weighted (avg) tree per sub-model
            if safe_isinstance(model, "pyspark.ml.classification.RandomForestClassificationModel") \
                    or safe_isinstance(model, "pyspark.ml.regression.RandomForestRegressionModel"):
                sum_weight = sum(model.treeWeights)  # output is average of trees
                self.trees = [SingleTree(tree, normalize=normalize, scaling=model.treeWeights[i]/sum_weight) for i, tree in enumerate(model.trees)]
            # Spark GBT, create 1 weighted (learning rate) tree per sub-model
            elif safe_isinstance(model, "pyspark.ml.classification.GBTClassificationModel") \
                    or safe_isinstance(model, "pyspark.ml.regression.GBTRegressionModel"):
                self.objective = "squared_error" # GBT subtree use the variance
                self.tree_output = "raw_value"
                self.trees = [SingleTree(tree, normalize=False, scaling=model.treeWeights[i]) for i, tree in enumerate(model.trees)]
            # Spark Basic model (single tree)
            elif safe_isinstance(model, "pyspark.ml.classification.DecisionTreeClassificationModel") \
                    or safe_isinstance(model, "pyspark.ml.regression.DecisionTreeRegressionModel"):
                self.trees = [SingleTree(model, normalize=normalize, scaling=1)]
            else:
                assert False, "Unsupported Spark model type: " + str(type(model))
        elif safe_isinstance(model, "xgboost.core.Booster"):
            import xgboost
            self.original_model = model
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            less_than_or_equal = False
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
            if xgb_loader.num_class > 0:
                self.num_stacked_models = xgb_loader.num_class
        elif safe_isinstance(model, "xgboost.sklearn.XGBClassifier"):
            import xgboost
            self.input_dtype = np.float32
            self.model_type = "xgboost"
            self.original_model = model.get_booster()
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            less_than_or_equal = False
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
            self.tree_limit = getattr(model, "best_ntree_limit", None)
            if xgb_loader.num_class > 0:
                self.num_stacked_models = xgb_loader.num_class
            if self.model_output == "predict_proba":
                if self.num_stacked_models == 1:
                    self.model_output = "probability_doubled" # with predict_proba we need to double the outputs to match
                else:
                    self.model_output = "probability"
        elif safe_isinstance(model, "xgboost.sklearn.XGBRegressor"):
            import xgboost
            self.original_model = model.get_booster()
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            less_than_or_equal = False
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
            self.tree_limit = getattr(model, "best_ntree_limit", None)
            if xgb_loader.num_class > 0:
                self.num_stacked_models = xgb_loader.num_class
        elif safe_isinstance(model, "xgboost.sklearn.XGBRanker"):
            import xgboost
            self.original_model = model.get_booster()
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            less_than_or_equal = False
            # Note: for ranker, leaving tree_output and objective as None as they
            # are not implemented in native code yet
            self.tree_limit = getattr(model, "best_ntree_limit", None)
            if xgb_loader.num_class > 0:
                self.num_stacked_models = xgb_loader.num_class
        elif safe_isinstance(model, "lightgbm.basic.Booster"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet

            self.objective = objective_name_map.get(model.params.get("objective", "regression"), None)
            self.tree_output = tree_output_name_map.get(model.params.get("objective", "regression"), None)

        elif safe_isinstance(model, "gpboost.basic.Booster"):
            assert_import("gpboost")
            self.model_type = "gpboost"
            self.original_model = model
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet

            self.objective = objective_name_map.get(model.params.get("objective", "regression"), None)
            self.tree_output = tree_output_name_map.get(model.params.get("objective", "regression"), None)

        elif safe_isinstance(model, "lightgbm.sklearn.LGBMRegressor"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
            if model.objective is None:
                self.objective = "squared_error"
                self.tree_output = "raw_value"
        elif safe_isinstance(model, "lightgbm.sklearn.LGBMRanker"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            # Note: for ranker, leaving tree_output and objective as None as they
            # are not implemented in native code yet
        elif safe_isinstance(model, "lightgbm.sklearn.LGBMClassifier"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
            if model.objective is None:
                self.objective = "binary_crossentropy"
                self.tree_output = "log_odds"
        elif safe_isinstance(model, "catboost.core.CatBoostRegressor"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
            self.cat_feature_indices = model.get_cat_feature_indices()
        elif safe_isinstance(model, "catboost.core.CatBoostClassifier"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
            self.input_dtype = np.float32
            try:
                cb_loader = CatBoostTreeModelLoader(model)
                self.trees = cb_loader.get_trees(data=data, data_missing=data_missing)
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.tree_output = "log_odds"
            self.objective = "binary_crossentropy"
            self.cat_feature_indices = model.get_cat_feature_indices()
        elif safe_isinstance(model, "catboost.core.CatBoost"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
            self.cat_feature_indices = model.get_cat_feature_indices()
        elif safe_isinstance(model, "imblearn.ensemble._forest.BalancedRandomForestClassifier"):
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif safe_isinstance(model, "ngboost.ngboost.NGBoost") or safe_isinstance(model, "ngboost.api.NGBRegressor") or safe_isinstance(model, "ngboost.api.NGBClassifier"):
            assert model.base_models, "The NGBoost model has empty `base_models`! Have you called `model.fit`?"
            if self.model_output == "raw":
                param_idx = 0 # default to the first parameter of the output distribution
                warnings.warn("Translating model_ouput=\"raw\" to model_output=0 for the 0-th parameter in the distribution. Use model_output=0 directly to avoid this warning.")
            elif type(self.model_output) is int:
                param_idx = self.model_output
                self.model_output = "raw" # note that after loading we have a new model_output type
            assert safe_isinstance(model.base_models[0][param_idx], ["sklearn.tree.DecisionTreeRegressor", "sklearn.tree.tree.DecisionTreeRegressor"]), "You must use default_tree_learner!"
            shap_trees = [trees[param_idx] for trees in model.base_models]
            self.internal_dtype = shap_trees[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = - model.learning_rate * np.array(model.scalings) # output is weighted average of trees
            self.trees = [SingleTree(e.tree_, scaling=s, data=data, data_missing=data_missing) for e,s in zip(shap_trees,scaling)]
            self.objective = objective_name_map.get(shap_trees[0].criterion, None)
            self.tree_output = "raw_value"
            self.base_offset = model.init_params[param_idx]
        else:
            self.trees = [SingleTree(model, normalize=True, data=data, data_missing=data_missing)]
            self.objective = "binary_crossentropy"
            self.tree_output = "probability"
            # raise Exception("Model type not yet supported by TreeExplainer: " + str(type(model)))

        # build a dense numpy version of all the tree objects
        if self.trees is not None and self.trees:
            max_nodes = np.max([len(t.values) for t in self.trees])
            assert len(np.unique([t.values.shape[1] for t in self.trees])) == 1, "All trees in the ensemble must have the same output dimension!"
            num_trees = len(self.trees)
            if self.num_stacked_models > 1:
                assert len(self.trees) % self.num_stacked_models == 0, "Only stacked models with equal numbers of trees are supported!"
                assert self.trees[0].values.shape[1] == 1, "Only stacked models with single outputs per model are supported!"
                self.num_outputs = self.num_stacked_models
            else:
                self.num_outputs = self.trees[0].values.shape[1]

            # important to be -1 in unused sections!! This way we can tell which entries are valid.
            self.children_left = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.children_right = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.children_default = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.features = -np.ones((num_trees, max_nodes), dtype=np.int32)

            self.thresholds = np.zeros((num_trees, max_nodes), dtype=self.internal_dtype)
            self.values = np.zeros((num_trees, max_nodes, self.num_outputs), dtype=self.internal_dtype)
            self.node_sample_weight = np.zeros((num_trees, max_nodes), dtype=self.internal_dtype)

            for i in range(num_trees):
                self.children_left[i,:len(self.trees[i].children_left)] = self.trees[i].children_left
                self.children_right[i,:len(self.trees[i].children_right)] = self.trees[i].children_right
                self.children_default[i,:len(self.trees[i].children_default)] = self.trees[i].children_default
                self.features[i,:len(self.trees[i].features)] = self.trees[i].features
                self.thresholds[i,:len(self.trees[i].thresholds)] = self.trees[i].thresholds
                if self.num_stacked_models > 1:
                    stack_pos = int(i // (num_trees / self.num_stacked_models))
                    self.values[i,:len(self.trees[i].values[:,0]),stack_pos] = self.trees[i].values[:,0]
                else:
                    self.values[i,:len(self.trees[i].values)] = self.trees[i].values
                self.node_sample_weight[i,:len(self.trees[i].node_sample_weight)] = self.trees[i].node_sample_weight

                # ensure that the passed background dataset lands in every leaf
                if np.min(self.trees[i].node_sample_weight) <= 0:
                    self.fully_defined_weighting = False

            # If we should do <= then we nudge the thresholds to make our <= work like <
            if not less_than_or_equal:
                self.thresholds = np.nextafter(self.thresholds, np.inf)

            self.num_nodes = np.array([len(t.values) for t in self.trees], dtype=np.int32)
            self.max_depth = np.max([t.max_depth for t in self.trees])

            # make sure the base offset is a 1D array
            if not hasattr(self.base_offset, "__len__") or len(self.base_offset) == 0:
                self.base_offset = (np.ones(self.num_outputs) * self.base_offset).astype(self.internal_dtype)
            self.base_offset = self.base_offset.flatten()
            assert len(self.base_offset) == self.num_outputs

class SingleTree:
    """ A single decision tree.
    The primary point of this object is to parse many different tree types into a common format.
    """
    def __init__(self, tree, normalize=False, scaling=1.0, data=None, data_missing=None):
        assert_import("cext")

        if safe_isinstance(tree, "sklearn.tree._tree.Tree"):
            self.children_left = tree.children_left.astype(np.int32)
            self.children_right = tree.children_right.astype(np.int32)
            self.children_default = self.children_left # missing values not supported in sklearn
            self.features = tree.feature.astype(np.int32)
            self.thresholds = tree.threshold.astype(np.float64)
            self.values = tree.value.reshape(tree.value.shape[0], tree.value.shape[1] * tree.value.shape[2])
            if normalize:
                self.values = (self.values.T / self.values.sum(1)).T
            self.values = self.values * scaling
            self.node_sample_weight = tree.weighted_n_node_samples.astype(np.float64)

        elif type(tree) is dict and 'features' in tree:
            self.children_left = tree["children_left"].astype(np.int32)
            self.children_right = tree["children_right"].astype(np.int32)
            self.children_default = tree["children_default"].astype(np.int32)
            self.features = tree["features"].astype(np.int32)
            self.thresholds = tree["thresholds"]
            self.values = tree["values"] * scaling
            self.node_sample_weight = tree["node_sample_weight"]

        # deprecated dictionary support (with sklearn singlular style "feature" and "value" names)
        elif type(tree) is dict and 'children_left' in tree:
            self.children_left = tree["children_left"].astype(np.int32)
            self.children_right = tree["children_right"].astype(np.int32)
            self.children_default = tree["children_default"].astype(np.int32)
            self.features = tree["feature"].astype(np.int32)
            self.thresholds = tree["threshold"]
            self.values = tree["value"] * scaling
            self.node_sample_weight = tree["node_sample_weight"]

        elif safe_isinstance(tree, "pyspark.ml.classification.DecisionTreeClassificationModel") \
                or safe_isinstance(tree, "pyspark.ml.regression.DecisionTreeRegressionModel"):
            #model._java_obj.numNodes() doesn't give leaves, need to recompute the size
            def getNumNodes(node, size):
                size = size + 1
                if node.subtreeDepth() == 0:
                    return size
                else:
                    size = getNumNodes(node.leftChild(), size)
                    return getNumNodes(node.rightChild(), size)

            num_nodes = getNumNodes(tree._java_obj.rootNode(), 0)
            self.children_left = np.full(num_nodes, -2, dtype=np.int32)
            self.children_right = np.full(num_nodes, -2, dtype=np.int32)
            self.children_default = np.full(num_nodes, -2, dtype=np.int32)
            self.features = np.full(num_nodes, -2, dtype=np.int32)
            self.thresholds = np.full(num_nodes, -2, dtype=np.float64)
            self.values = [-2]*num_nodes
            self.node_sample_weight = np.full(num_nodes, -2, dtype=np.float64)
            def buildTree(index, node):
                index = index + 1
                if tree._java_obj.getImpurity() == 'variance':
                    self.values[index] = [node.prediction()] #prediction for the node
                else:
                    self.values[index] = [e for e in node.impurityStats().stats()] #for gini: NDarray(numLabel): 1 per label: number of item for each label which went through this node
                self.node_sample_weight[index] = node.impurityStats().count() #weighted count of element trough this node

                if node.subtreeDepth() == 0:
                    return index
                else:
                    self.features[index] = node.split().featureIndex() #index of the feature we split on, not available for leaf, int
                    if str(node.split().getClass()).endswith('tree.CategoricalSplit'):
                        #Categorical split isn't implemented, TODO: could fake it by creating a fake node to split on the exact value?
                        raise NotImplementedError('CategoricalSplit are not yet implemented')
                    self.thresholds[index] = node.split().threshold() #threshold for the feature, not available for leaf, float

                    self.children_left[index] = index + 1
                    idx = buildTree(index, node.leftChild())
                    self.children_right[index] = idx + 1
                    idx = buildTree(idx, node.rightChild())
                    return idx

            buildTree(-1, tree._java_obj.rootNode())
            #default Not supported with mlib? (TODO)
            self.children_default = self.children_left
            self.values = np.asarray(self.values)
            if normalize:
                self.values = (self.values.T / self.values.sum(1)).T
            self.values = self.values * scaling

        elif type(tree) == dict and 'tree_structure' in tree: # LightGBM model dump
            start = tree['tree_structure']
            num_parents = tree['num_leaves']-1
            self.children_left = np.empty((2*num_parents+1), dtype=np.int32)
            self.children_right = np.empty((2*num_parents+1), dtype=np.int32)
            self.children_default = np.empty((2*num_parents+1), dtype=np.int32)
            self.features = np.empty((2*num_parents+1), dtype=np.int32)
            self.thresholds = np.empty((2*num_parents+1), dtype=np.float64)
            self.values = [-2]*(2*num_parents+1)
            self.node_sample_weight = np.empty((2*num_parents+1), dtype=np.float64)
            visited, queue = [], [start]
            while queue:
                vertex = queue.pop(0)
                if 'split_index' in vertex.keys():
                    if vertex['split_index'] not in visited:
                        if 'split_index' in vertex['left_child'].keys():
                            self.children_left[vertex['split_index']] = vertex['left_child']['split_index']
                        else:
                            self.children_left[vertex['split_index']] = vertex['left_child']['leaf_index']+num_parents
                        if 'split_index' in vertex['right_child'].keys():
                            self.children_right[vertex['split_index']] = vertex['right_child']['split_index']
                        else:
                            self.children_right[vertex['split_index']] = vertex['right_child']['leaf_index']+num_parents
                        if vertex['default_left']:
                            self.children_default[vertex['split_index']] = self.children_left[vertex['split_index']]
                        else:
                            self.children_default[vertex['split_index']] = self.children_right[vertex['split_index']]
                        self.features[vertex['split_index']] = vertex['split_feature']
                        self.thresholds[vertex['split_index']] = vertex['threshold']
                        self.values[vertex['split_index']] = [vertex['internal_value']]
                        self.node_sample_weight[vertex['split_index']] = vertex['internal_count']
                        visited.append(vertex['split_index'])
                        queue.append(vertex['left_child'])
                        queue.append(vertex['right_child'])
                else:
                    self.children_left[vertex['leaf_index']+num_parents] = -1
                    self.children_right[vertex['leaf_index']+num_parents] = -1
                    self.children_default[vertex['leaf_index']+num_parents] = -1
                    self.features[vertex['leaf_index']+num_parents] = -1
                    self.children_left[vertex['leaf_index']+num_parents] = -1
                    self.children_right[vertex['leaf_index']+num_parents] = -1
                    self.children_default[vertex['leaf_index']+num_parents] = -1
                    self.features[vertex['leaf_index']+num_parents] = -1
                    self.thresholds[vertex['leaf_index']+num_parents] = -1
                    self.values[vertex['leaf_index']+num_parents] = [vertex['leaf_value']]
                    self.node_sample_weight[vertex['leaf_index']+num_parents] = vertex['leaf_count']
            self.values = np.asarray(self.values)
            self.values = np.multiply(self.values, scaling)

        elif type(tree) == dict and 'nodeid' in tree:
            """ Directly create tree given the JSON dump (with stats) of a XGBoost model.
            """

            def max_id(node):
                if "children" in node:
                    return max(node["nodeid"], *[max_id(n) for n in node["children"]])
                else:
                    return node["nodeid"]

            m = max_id(tree) + 1
            self.children_left = -np.ones(m, dtype=np.int32)
            self.children_right = -np.ones(m, dtype=np.int32)
            self.children_default = -np.ones(m, dtype=np.int32)
            self.features = -np.ones(m, dtype=np.int32)
            self.thresholds = np.zeros(m, dtype=np.float64)
            self.values = np.zeros((m, 1), dtype=np.float64)
            self.node_sample_weight = np.empty(m, dtype=np.float64)

            def extract_data(node, tree):
                i = node["nodeid"]
                tree.node_sample_weight[i] = node["cover"]

                if "children" in node:
                    tree.children_left[i] = node["yes"]
                    tree.children_right[i] = node["no"]
                    tree.children_default[i] = node["missing"]
                    tree.features[i] = node["split"]
                    tree.thresholds[i] = node["split_condition"]

                    for n in node["children"]:
                        extract_data(n, tree)
                elif "leaf" in node:
                    tree.values[i] = node["leaf"] * scaling

            extract_data(tree, self)

        elif type(tree) == str:
            """ Build a tree from a text dump (with stats) of xgboost.
            """

            nodes = [t.lstrip() for t in tree[:-1].split("\n")]
            nodes_dict = {}
            for n in nodes: nodes_dict[int(n.split(":")[0])] = n.split(":")[1]
            m = max(nodes_dict.keys())+1
            children_left = -1*np.ones(m,dtype="int32")
            children_right = -1*np.ones(m,dtype="int32")
            children_default = -1*np.ones(m,dtype="int32")
            features = -2*np.ones(m,dtype="int32")
            thresholds = -1*np.ones(m,dtype="float64")
            values = 1*np.ones(m,dtype="float64")
            node_sample_weight = np.zeros(m,dtype="float64")
            values_lst = list(nodes_dict.values())
            keys_lst = list(nodes_dict.keys())
            for i in range(0,len(keys_lst)):
                value = values_lst[i]
                key = keys_lst[i]
                if ("leaf" in value):
                    # Extract values
                    val = float(value.split("leaf=")[1].split(",")[0])
                    node_sample_weight_val = float(value.split("cover=")[1])
                    # Append to lists
                    values[key] = val
                    node_sample_weight[key] = node_sample_weight_val
                else:
                    c_left = int(value.split("yes=")[1].split(",")[0])
                    c_right = int(value.split("no=")[1].split(",")[0])
                    c_default = int(value.split("missing=")[1].split(",")[0])
                    feat_thres = value.split(" ")[0]
                    if ("<" in feat_thres):
                        feature = int(feat_thres.split("<")[0][2:])
                        threshold = float(feat_thres.split("<")[1][:-1])
                    if ("=" in feat_thres):
                        feature = int(feat_thres.split("=")[0][2:])
                        threshold = float(feat_thres.split("=")[1][:-1])
                    node_sample_weight_val = float(value.split("cover=")[1].split(",")[0])
                    children_left[key] = c_left
                    children_right[key] = c_right
                    children_default[key] = c_default
                    features[key] = feature
                    thresholds[key] = threshold
                    node_sample_weight[key] = node_sample_weight_val

            self.children_left = children_left
            self.children_right = children_right
            self.children_default = children_default
            self.features = features
            self.thresholds = thresholds
            self.values = values[:,np.newaxis] * scaling
            self.node_sample_weight = node_sample_weight
        else:
            num_nodes = tree._decision_node_cnt + tree._active_leaf_node_cnt
            # print(num_nodes)
            # print(tree.get_model_description())
            self.children_left = np.full(num_nodes, -2, dtype=np.int32)
            self.children_right = np.full(num_nodes, -2, dtype=np.int32)
            self.children_default = np.full(num_nodes, -2, dtype=np.int32)
            self.features = np.full(num_nodes, -2, dtype=np.int32)
            self.thresholds = np.full(num_nodes, -2, dtype=np.float64)
            self.values = [-2]*num_nodes
            self.node_sample_weight = np.full(num_nodes, -2, dtype=np.float64)
            def buildTree(index, node):
                # print(node)
                # exit()
                index = index + 1
                # self.values[index] = [int(e) for e in node._observed_class_distribution.values()] #for gini: NDarray(numLabel): 1 per label: number of item for each label which went through this node
                # print(tree.classes)
                self.values[index] = [int(node._observed_class_distribution[e]) if e in node._observed_class_distribution else 0 for e in tree.classes] #for gini: NDarray(numLabel): 1 per label: number of item for each label which went through this node
                self.node_sample_weight[index] = sum(node._observed_class_distribution.values()) #weighted count of element trough this node

                if node.subtree_depth() == 0:
                    return index
                else:
                    self.features[index] = node._split_test._att_idx #index of the feature we split on, not available for leaf, int
                    self.thresholds[index] = node._split_test._att_value #threshold for the feature, not available for leaf, float

                    self.children_left[index] = index + 1
                    idx = buildTree(index, node.get_child(0))
                    self.children_right[index] = idx + 1
                    idx = buildTree(idx, node.get_child(1))
                    return idx

            buildTree(-1, tree._tree_root)
            #default Not supported with mlib? (TODO)
            self.children_default = self.children_left
            self.values = np.asarray(self.values)
            if normalize:
                self.values = (self.values.T / self.values.sum(1)).T
            self.values = self.values * scaling
            # raise Exception("Unknown input to SingleTree constructor: " + str(tree))

        # Re-compute the number of samples that pass through each node if we are given data
        if data is not None and data_missing is not None:
            self.node_sample_weight[:] = 0.0
            _cext.dense_tree_update_weights(
                self.children_left, self.children_right, self.children_default, self.features,
                self.thresholds, self.values, 1, self.node_sample_weight, data, data_missing
            )

        # we compute the expectations to make sure they follow the SHAP logic
        self.max_depth = _cext.compute_expectations(
            self.children_left, self.children_right, self.node_sample_weight,
            self.values
        )