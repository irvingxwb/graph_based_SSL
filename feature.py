from collections import Counter
from helper import operate_dict, word2features
from crf import processLabelData
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics import pairwise_distances
import timeit
import logging
import math
import time

logger = logging.getLogger("FeatureSet")

STARTING_LABEL = '*'  # Label of t=-1
STARTING_LABEL_INDEX = 0


def default_feature_func(_, sent, i):
    feature = word2features(sent, i)
    ret = []
    for key, item in feature.items():
        feature_string = key + str(item)
        ret.append(feature_string)

    return ret


class FeatureSet():
    feature_dic = dict()
    observation_set = set()
    empirical_counts = Counter()
    num_features = 0
    feature_func = default_feature_func

    label_dic = {STARTING_LABEL: STARTING_LABEL_INDEX}
    label_array = [STARTING_LABEL]

    def __init__(self, data_set):
        # X is ngrams vector and Y is its realated feature vector
        label_X, label_Y = processLabelData(data_set)
        for X, Y in zip(label_X, label_Y):
            prev_y = STARTING_LABEL_INDEX
            if len(X) != len(Y):
                print(X, Y)
            for t in range(len(X)):
                # Gets a label id
                if Y[t] not in self.label_dic:
                    y = len(self.label_dic)
                    self.label_dic[Y[t]] = y
                    self.label_array.append(Y[t])
                # Adds features
                self._add(prev_y, y, X, t)
                prev_y = y

    def __len__(self):
        return self.num_features

    def _add(self, prev_y, y, X, t):
        for feature_string in self.feature_func(X, t):
            if feature_string in self.feature_dic.keys():
                if (prev_y, y) in self.feature_dic[feature_string].keys():
                    self.empirical_counts[self.feature_dic[feature_string][(prev_y, y)]] += 1
                else:
                    feature_id = self.num_features
                    self.feature_dic[feature_string][(prev_y, y)] = feature_id
                    self.empirical_counts[feature_id] += 1
                    self.num_features += 1
            # Unigram feature
            #     if (-1, y) in self.feature_dic[feature_string].keys():
            #         self.empirical_counts[self.feature_dic[feature_string][(-1, y)]] += 1
            #     else:
            #         feature_id = self.num_features
            #         self.feature_dic[feature_string][(-1, y)] = feature_id
            #         self.empirical_counts[feature_id] += 1
            #         self.num_features += 1
            else:
                self.feature_dic[feature_string] = dict()
                # Bigram feature
                feature_id = self.num_features
                self.feature_dic[feature_string][(prev_y, y)] = feature_id
                self.empirical_counts[feature_id] += 1
                self.num_features += 1
                # Unigram feature
                # feature_id = self.num_features
                # self.feature_dic[feature_string][(-1, y)] = feature_id
                # self.empirical_counts[feature_id] += 1
                # self.num_features += 1

    def calc_inner_products(self, params, X, t):
        inner_products = Counter()
        for feature_string in self.feature_func(X, t):
            try:
                for (prev_y, y), feature_id in self.feature_dic[feature_string].items():
                    inner_products[(prev_y, y)] += params[feature_id]
            except KeyError:
                print("KeyError for calc inner product")
                pass
        return [((prev_y, y), score) for (prev_y, y), score in inner_products.items()]

    def get_labels(self):
        return self.label_dic, self.label_array

    def get_empirical_counts(self):
        empirical_counts = np.ndarray((self.num_features,))
        for feature_id, counts in self.empirical_counts.items():
            empirical_counts[feature_id] = counts
        return empirical_counts

    def get_feature_list(self, X, t):
        feature_list_dic = dict()
        for feature_string in self.feature_func(X, t):
            for (prev_y, y), feature_id in self.feature_dic[feature_string].items():
                if (prev_y, y) in feature_list_dic.keys():
                    feature_list_dic[(prev_y, y)].add(feature_id)
                else:
                    feature_list_dic[(prev_y, y)] = {feature_id}
        return [((prev_y, y), feature_ids) for (prev_y, y), feature_ids in feature_list_dic.items()]




