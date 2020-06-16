import logging
import numpy as np
from math import log, exp
from scipy.optimize import fmin_l_bfgs_b
from collections import Counter

logger = logging.getLogger("LinearChainCRF")

SCALING_THRESHOLD = 1e250

ITERATION_NUM = 0
SUB_ITERATION_NUM = 0
TOTAL_SUB_ITERATIONS = 0
STARTING_LABEL_INDEX = 0
GRADIENT = None


def _callback(params):
    # global ITERATION_NUM
    # global SUB_ITERATION_NUM
    # global TOTAL_SUB_ITERATIONS
    # ITERATION_NUM += 1
    # TOTAL_SUB_ITERATIONS += SUB_ITERATION_NUM
    # SUB_ITERATION_NUM = 0
    pass


def _forward_backward(num_labels, time_length, potential_table):
    alpha = np.zeros((time_length, num_labels))
    scaling_dic = dict()
    t = 0
    for label_id in range(num_labels):
        alpha[t, label_id] = potential_table[t][STARTING_LABEL_INDEX, label_id]

    # alpha[0, :] = potential_table[0][STARTING_LABEL_INDEX, :]  # slow
    t = 1
    while t < time_length:
        scaling_time = None
        scaling_coefficient = None
        overflow_occured = False
        label_id = 1
        while label_id < num_labels:
            alpha[t, label_id] = np.dot(alpha[t - 1, :], potential_table[t][:, label_id])
            if alpha[t, label_id] > SCALING_THRESHOLD:
                if overflow_occured:
                    raise BaseException()
                overflow_occured = True
                scaling_time = t - 1
                scaling_coefficient = SCALING_THRESHOLD
                scaling_dic[scaling_time] = scaling_coefficient
                break
            else:
                label_id += 1
        if overflow_occured:
            alpha[t - 1] /= scaling_coefficient
            alpha[t] = 0
        else:
            t += 1

    beta = np.zeros((time_length, num_labels))
    t = time_length - 1
    for label_id in range(num_labels):
        beta[t, label_id] = 1.0

    for t in range(time_length - 2, -1, -1):
        for label_id in range(1, num_labels):
            beta[t, label_id] = np.dot(beta[t + 1, :], potential_table[t + 1][label_id, :])
        if t in scaling_dic.keys():
            beta[t] /= scaling_dic[t]

    Z = sum(alpha[time_length - 1])

    return alpha, beta, Z, scaling_dic


def _generate_potential_table(params, num_labels, feature_set, X, inference=True):
    """
    Generates a potential table using given observations.
    * potential_table[t][prev_y, y]
        := exp(inner_product(params, feature_vector(prev_y, y, X, t)))
        (where 0 <= t < len(X))
    """
    tables = list()
    for t in range(len(X)):
        table = np.zeros((num_labels, num_labels))
        if inference:
            for (prev_y, y), score in feature_set.calc_inner_products(params, X, t):
                if prev_y == -1:
                    table[:, y] += score
                else:
                    table[prev_y, y] += score
        else:
            for (prev_y, y), feature_ids in X[t]:
                score = sum(params[fid] for fid in feature_ids)
                if prev_y == -1:
                    table[:, y] += score
                else:
                    table[prev_y, y] += score
        table = np.exp(table)
        if t == 0:
            table[STARTING_LABEL_INDEX + 1:] = 0
        else:
            table[:, STARTING_LABEL_INDEX] = 0
            table[STARTING_LABEL_INDEX, :] = 0
        tables.append(table)

    return tables


def _log_likelihood(params, *args):
    feature_set, training_feature_data, empirical_counts, label_dic, squared_sigma = args
    expected_counts = np.zeros(len(feature_set))

    total_logZ = 0
    for X_features in training_feature_data:
        potential_table = _generate_potential_table(params, len(label_dic), feature_set,
                                                    X_features, inference=False)
        alpha, beta, Z, scaling_dic = _forward_backward(len(label_dic), len(X_features), potential_table)
        total_logZ += log(Z) + \
                      sum(log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())
        for t in range(len(X_features)):
            potential = potential_table[t]
            for (prev_y, y), feature_ids in X_features[t]:
                # Adds p(prev_y, y | X, t)
                if prev_y == -1:
                    if t in scaling_dic.keys():
                        prob = (alpha[t, y] * beta[t, y] * scaling_dic[t]) / Z
                    else:
                        prob = (alpha[t, y] * beta[t, y]) / Z
                elif t == 0:
                    if prev_y is not STARTING_LABEL_INDEX:
                        continue
                    else:
                        prob = (potential[STARTING_LABEL_INDEX, y] * beta[t, y]) / Z
                else:
                    if prev_y is STARTING_LABEL_INDEX or y is STARTING_LABEL_INDEX:
                        continue
                    else:
                        prob = (alpha[t - 1, prev_y] * potential[prev_y, y] * beta[t, y]) / Z
                for fid in feature_ids:
                    expected_counts[fid] += prob

    likelihood = np.dot(empirical_counts, params) - total_logZ - \
                 np.sum(np.dot(params, params)) / (squared_sigma * 2)

    gradients = empirical_counts - expected_counts - params / squared_sigma
    global GRADIENT
    GRADIENT = gradients

    # global SUB_ITERATION_NUM
    # sub_iteration_str = '    '
    # if SUB_ITERATION_NUM > 0:
    #     sub_iteration_str = '(' + '{0:02d}'.format(SUB_ITERATION_NUM) + ')'
    # print('  ', '{0:03d}'.format(ITERATION_NUM), sub_iteration_str, ':', likelihood * -1)
    #
    # SUB_ITERATION_NUM += 1

    return likelihood * -1


def _gradient(params, *args):
    return GRADIENT * -1


def _get_training_feature_data(self):
    return [[self.feature_set.get_feature_list(X, t) for t in range(len(X))]
            for X, _ in self.training_data]


class LinearChainCRF():
    training_data = None
    feature_set = None

    label_dic = None
    label_array = None
    num_labels = None

    params = np.zeros(len(feature_set))
    # For L-BFGS
    squared_sigma = 10.0

    def __init__(self, feature_set, training_data):
        self.feature_set = feature_set
        self.training_data = training_data

    def train(self, corpus_filename, model_filename):
        logger.debug('Start training CRF')

        self.label_dic, self.label_array = self.feature_set.get_labels()
        self.num_labels = len(self.label_array)
        logger.debug("** Number of labels: %d" % (self.num_labels - 1))
        logger.debug("** Number of features: %d" % len(self.feature_set))

        # Estimates parameters to maximize log-likelihood of the corpus.
        self._estimate_parameters()

        logger.debug('Training done')

    def _estimate_parameters(self):
        training_feature_data = self._get_training_feature_data()
        logger.debug('* Estimate parameters start L-BGFS')

        for X_features in training_feature_data:
            print(X_features)
            logger.debug('* start compute potential table and forward backward')
            potential_table = _generate_potential_table(self.params, self.num_labels, self.feature_set,
                                                        X_features, inference=False)
            alpha, beta, Z, scaling_dic = _forward_backward(self.num_labels, len(X_features), potential_table)
            logger.debug('* end computing')
            break

        # self.params, log_likelihood, information = \
        #     fmin_l_bfgs_b(func=_log_likelihood, fprime=_gradient,
        #                   x0=np.zeros(len(self.feature_set)),
        #                   args=(self.feature_set, training_feature_data,
        #                         self.feature_set.get_empirical_counts(),
        #                         self.label_dic, self.squared_sigma),
        #                   callback=_callback)
        logger.debug('* Training has been finished')

        # if information['warnflag'] != 0:
        #     print('* Warning (code: %d)' % information['warnflag'])
        #     if 'task' in information.keys():
        #         print('* Reason: %s' % (information['task']))
        # print('* Likelihood: %s' % str(log_likelihood))
