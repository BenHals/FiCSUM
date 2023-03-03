import math
import logging
from collections import Counter, deque
import itertools
from copy import deepcopy

from skmultiflow.utils import get_dimensions, check_random_state
from skmultiflow.drift_detection.adwin import ADWIN
import numpy as np
import scipy.stats
import traceback

import math
import statistics
from collections import deque
import math

import scipy.stats
import numpy as np


def make_detector(warn=False, s=1e-5):
    sensitivity = s * 2 if warn else s
    return ADWIN(delta=sensitivity)


class state:
    def __init__(self, id, learner):
        self.id = id
        self.classifier = learner
        self.seen = 0
        self.current_evolution = self.classifier.evolution

    def observe(self):
        self.seen += 1

    def transition(self):
        pass

    def start_evolution_transition(self, dirty_length):
        self.current_evolution = self.classifier.evolution

    def __str__(self):
        return f"<State {self.id}>"

    def __repr__(self):
        return self.__str__()


class CDClassifier:
    def __init__(self,
                 suppress=False,
                 learner=None,
                 sensitivity=0.05,
                 poisson=6):

        if learner is None:
            raise ValueError('Need a learner')

        # learner is the classifier used by each state.
        # papers use HoeffdingTree from scikit-multiflow
        self.learner = learner

        # sensitivity is the sensitivity of the concept
        # drift detector
        self.sensitivity = sensitivity
        self.base_sensitivity = sensitivity
        self.current_sensitivity = sensitivity

        # suppress debug info
        self.suppress = suppress

        # rand_weights is if a strategy is setting sample
        # weights for training
        self.rand_weights = poisson > 1

        # poisson is the strength of sample weighting
        # based on leverage bagging
        self.poisson = poisson

        # setup initial drift detectors
        self.detector = make_detector(s=sensitivity)
        self.warn_detector = make_detector(
            s=self.get_warning_sensitivity(sensitivity))
        self.in_warning = False
        self.last_warning_point = 0
        self.warning_detected = False

        # initialize waiting state. If we don't have enough
        # data to select the next concept, we wait until we do.
        self.waiting_for_concept_data = False

        # init the current number of states
        self.max_state_id = 0

        # init randomness
        self.random_state = None
        self._random_state = check_random_state(self.random_state)

        self.ex = -1
        self.classes = None
        self._train_weight_seen_by_model = 0

        # init data which is exposed to evaluators
        self.found_change = False
        self.num_states = 1
        self.active_state = self.max_state_id
        self.states = []

        self.detected_drift = False

        # track the last predicted label
        self.last_label = 0

        # set up repository
        self.state_repository = {}
        init_id = self.max_state_id
        self.max_state_id += 1
        init_state = state(init_id, self.learner())
        self.state_repository[init_id] = init_state
        self.active_state_id = init_id

        self.manual_control = False
        self.force_transition = False
        self.force_transition_only = False
        self.force_learn_fingerprint = False
        self.force_stop_learn_fingerprint = False
        self.force_transition_to = None
        self.force_transition_to = None
        self.force_lock_weights = False
        self.force_locked_weights = None
        self.force_stop_fingerprint_age = None
        self.force_stop_add_to_normalizer = False

    def get_warning_sensitivity(self, s):
        return s * 2

    def get_active_state(self):
        return self.state_repository[self.active_state_id]

    def make_state(self):
        new_id = self.max_state_id
        self.max_state_id += 1
        return new_id, state(new_id, self.learner())

    def reset(self):
        pass

    def get_temporal_x(self, X):
        # return np.concatenate([X, self.last_label], axis = None)
        return np.concatenate([X], axis=None)

    def predict(self, X):
        """
        Predict using the model of the currently active state.
        """
        logging.debug("Predicting")
        # ll = self.last_label if last_label is None else last_label
        temporal_X = self.get_temporal_x(X)
        logging.debug(f"temporal_X: {temporal_X}")

        # ll = 0
        return self.get_active_state().classifier.predict([temporal_X])

    def partial_fit(self, X, y, classes=None, sample_weight=None, masked=False):
        """
        Fit an array of observations.
        Splits input into individual observations and
        passes to a helper function _partial_fit.
        Randomly weights observations depending on 
        Config.
        """

        if masked:
            return
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError(
                    'Inconsistent number of instances ({}) and weights ({}).'
                    .format(row_cnt, len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._train_weight_seen_by_model += sample_weight[i]
                    self.ex += 1
                    if self.rand_weights and self.poisson >= 1:
                        k = self.poisson
                        sample_weight[i] = k
                    self._partial_fit(X[i], y[i], sample_weight[i], masked)

    def get_imputed_label(self, X, prediction, last_label):
        """ Get a label.
        Imputes when the true label is masked
        """

        return prediction

    def _partial_fit(self, X, y, sample_weight, masked=False):
        self.warning_detected = False
        logging.debug(
            f"Partial fit on X: {X}, y:{y}, masked: {masked}, using state {self.active_state_id}")

        # get_temporal_x, and get_imputed_label are to deal with masked
        # values where we don't see true label.
        # As the functions are now, they don't do anything extra.
        # But could be extended to reuse last made prediction as
        # the label for example.
        temporal_X = self.get_temporal_x(X)
        np.seterr(all='ignore')
        prediction = self.predict(temporal_X)[0]

        label = y if not masked else self.get_imputed_label(
            X=X, prediction=prediction, last_label=self.last_label)
        self.last_label = label

        # correctly_classified from the systems point of view.
        correctly_classifies = prediction == label

        # init defaults for trackers
        found_change = False
        self.detected_drift = False
        current_sensitivity = self.get_current_sensitivity()

        fit = True
        fit = not self.force_stop_learn_fingerprint
        # Fit the classifier of the current state.
        # We fit on buffered values, to avoid
        # fitting on items from a dufferent concept

        fit = fit
        if fit:
            logging.debug(f"Fitting state {self.active_state_id} at {self.ex}")
            self.get_active_state().classifier.partial_fit(
                np.asarray([temporal_X]),
                np.asarray([label]),
                sample_weight=np.asarray([sample_weight]),
                classes=self.classes
            )

        # Detect if the fit changed the model in a way which
        # might significantly change behaviour.
        classifier_evolved = self.get_active_state(
        ).classifier.evolution > self.get_active_state().current_evolution

        # LOGGING
        if self.get_active_state().classifier.evolution < self.get_active_state().current_evolution:
            logging.debug(
                "ALERT: Current evolution less than classifier evolution. SHould not happen")

        # If we evolved, we reclassify items in our buffer
        # This is purely for training purposes, as we have
        # already emitted our prediction for these values.
        # These have not been used to fit yet, so this is not biased.
        # This captures the current behaviour of the model.
        # We evolve our state, or reset plasticity of performance
        # features so they can be molded to the new behaviour.
        if classifier_evolved:
            logging.debug(f"Classifier Evolved")
            self.get_active_state().start_evolution_transition(1)

        # For logging
        self.monitor_active_state_active_similarity = None
        self.monitor_active_state_buffered_similarity = None
        self.monitor_all_state_buffered_similarity = None
        self.monitor_all_state_active_similarity = None
        # logging.debug(f"{self.manual_control=}")
        # logging.debug(f"{self.force_learn_fingerprint=}")
        # logging.debug(f"{self.force_stop_learn_fingerprint=}")

        take_measurement = True
        if self.manual_control:
            take_measurement = self.force_learn_fingerprint and not self.force_stop_learn_fingerprint

        if take_measurement:
            logging.debug(
                f"Taking similarity measurement {self.active_state_id} at {self.ex}")
            # logging.debug(f"{take_measurement=}")
            logging.info(
                f"Adding {correctly_classifies} to change detector at {self.ex}")
            # Add error to detectors
            self.detector.delta = current_sensitivity
            self.warn_detector.delta = self.get_warning_sensitivity(
                current_sensitivity)
            self.detector.add_element(int(correctly_classifies))
            self.warn_detector.add_element(int(correctly_classifies))

            # For logging
            self.monitor_active_state_active_similarity = correctly_classifies
            self.similarity_last_observation = self.ex

        # If the warning detector fires we record the position
        # and reset. We take the most recent warning as the
        # start of out window, assuming this warning period
        # contains elements of the new state.
        if self.warn_detector.detected_change():
            logging.info(f"Detected warning at {self.ex}")
            self.warning_detected = False
            self.in_warning = True
            self.last_warning_point = self.ex
            self.warn_detector = make_detector(
                s=self.get_warning_sensitivity(current_sensitivity))

        if not self.in_warning:
            self.last_warning_point = max(0, self.ex - 100)

        # If the main state trigers, or we held off on changing due to lack of data,
        # trigger a change
        detected_drift = self.detector.detected_change()
        found_change = detected_drift or self.waiting_for_concept_data
        if found_change:
            logging.info(f"Detected Change at {self.ex}")

        # Just for monitoring
        self.detected_drift = detected_drift or self.force_transition

        if self.manual_control or self.force_transition_only:
            found_change = self.force_transition
            self.trigger_transition_check = False
            self.trigger_attempt_transition_check = False

        if found_change:
            logging.info(f"Found Change at {self.ex}")
            self.in_warning = False

            # Find the inactive models most suitable for the current stream. Also return a shadow model
            # trained on the warning period.
            # If none of these have high accuracy, hold off the adaptation until we have the data.
            ranked_alternatives, use_shadow, shadow_model, can_find_good_model = (
                [], True, self.learner(), True)
            logging.debug(f"Candidates: {ranked_alternatives}")
            forced_transition = False
            if self.manual_control and self.force_transition_to is not None:
                forced_transition = True
                logging.debug(
                    f"Forced transition to {self.force_transition_to}")
                if self.force_transition_to in self.state_repository:
                    ranked_alternatives = [self.force_transition_to]
                    use_shadow = False
                    shadow_model = None
                    can_find_good_model = True
                else:
                    ranked_alternatives = []
                    use_shadow = True
                    shadow_state = state(
                        self.force_transition_to, self.learner())
                    shadow_model = shadow_state.classifier
                    can_find_good_model = True

            if can_find_good_model:

                # we need to reset if this is a real transition.
                need_to_reset = use_shadow or (
                    ranked_alternatives[-1] != self.active_state_id)

                if need_to_reset:
                    self.get_active_state().transition()
                # If we determined the shadow is the best model, we mark it as a new state,
                # copy the model trained on the warning period across and set as the active state
                if use_shadow:
                    logging.info(f"Transition to shadow")
                    self.active_state_is_new = True
                    shadow_id, shadow_state = self.make_state()
                    shadow_state.classifier = shadow_model
                    self.state_repository[shadow_id] = shadow_state
                    self.active_state_id = shadow_id
                else:
                    # Otherwise we just set the found state to be active
                    logging.info(f"Transition to {ranked_alternatives[-1]}")
                    self.active_state_is_new = False
                    transition_target_id = ranked_alternatives[-1]
                    self.active_state_id = transition_target_id

                if need_to_reset:
                    # We reset drift detection as the new performance will not
                    # be the same, and reset history for the new state.
                    self.get_active_state().transition()
                    self.waiting_for_concept_data = False
                    self.detector = make_detector(s=current_sensitivity)
                    self.warn_detector = make_detector(
                        s=self.get_warning_sensitivity(current_sensitivity))

            else:
                # If we did not have enough data to find any good concepts to
                # transition to, wait until we do.
                self.waiting_for_concept_data = True

        # Set exposed info for evaluation.
        self.active_state = self.active_state_id
        self.found_change = found_change
        self.states = self.state_repository
        self.current_sensitivity = current_sensitivity

    def get_current_sensitivity(self):
        return self.base_sensitivity
