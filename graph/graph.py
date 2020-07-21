from .functions import operate_dict
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics import pairwise_distances
from collections import Counter
import numpy as np
import timeit
import logging
import math
import torch

logger = logging.getLogger("Graph")


class Graph:
    data_set = None

    # pmi
    feature_dict = {}
    ngrams_counter = None
    feature_counters = None
    ngrams_feature_counters = None
    ngrams_feature_map = None

    sum_features = 0
    sum_ngrams = 0
    pmi_vectors = None

    # graph
    ngrams = None
    graph_map = None
    ngram_prob_map = None

    # k for k_nearest
    def __init__(self, data_set):
        self.data_set = data_set
        self.build_feature_dicts()

        self.ngrams = self.ngrams_feature_map.keys()
        self.graph_map = dict.fromkeys(self.ngrams_feature_map.keys())
        self.length = len(self.ngrams)
        # compute k nearest map

    def __len__(self):
        return self.length

    def build_feature_dicts(self):
        # self.features = set()
        ngrams_list, features_list = self.data_set.get_graph_list()

        keys = features_list[0].keys()
        feature_agg = []
        ngrams_feature_agg = []

        self.ngrams_counter = Counter(ngrams_list)
        # count feature numbers
        self.feature_counters = {}
        # count (ngram, feature) numbers
        self.ngrams_feature_counters = {}
        # each ngram got different featurs
        self.ngrams_feature_map = {}

        # gather all features by its name
        feature_count = 0
        for ngram, features in zip(ngrams_list, features_list):
            if ngram not in self.ngrams_feature_map:
                self.ngrams_feature_map[ngram] = [features]
            else:
                self.ngrams_feature_map[ngram].append(features)

            for feature_name, feature in features.items():
                feature_agg.append(feature)
                ngrams_feature_agg.append((ngram, feature))

                if feature not in self.feature_dict:
                    self.feature_dict[feature] = feature_count
                    feature_count += 1

        self.feature_counters = Counter(feature)
        self.ngrams_feature_counters = Counter(self.ngram_feature)

        # self.features = list(self.features)
        self.sum_features = len(self.feature_dict)
        self.sum_ngrams = len(self.ngrams_feature_map)

        logger.debug("PMI: complete pmi with: %s %s" % str(self.sum_ngrams), str(self.sum_features))

    def build_pmi_vectors(self):
        ngram_idx = 0
        # self.pmi_vectors = lil_matrix((self.sum_ngrams, self.sum_features))
        # for ngram, features in self.ngrams_feature_map.items():
        #     for feature in features:
        #         for name, item in feature.items():
        #             item_idx = self.feature_dict[item]
        #             self.pmi_vectors[ngram_idx, item_idx] = self.pmi_score(ngram, item, name)
        #
        #     ngram_idx += 1

        self.pmi_vectors = torch.zeros(self.sum_ngrams, self.sum_features, dtype=torch.float)
        for ngram, features in self.ngrams_feature_map.items():
            for feature in features:
                for name, item in feature.items():
                    item_idx = self.feature_dict[item]
                    self.pmi_vectors[ngram_idx, item_idx] = self.pmi_score(ngram, item, name)

            ngram_idx += 1

    def pmi_score(self, ngram, feature, feature_name):

        count_ngram_feature = self.ngrams_feature_counters[feature_name][(ngram, feature)]
        count_ngram = self.ngrams_counter[ngram]
        count_feature = self.feature_counters[feature_name][feature]

        score = np.log((count_ngram_feature * self.sum_ngrams) / (count_ngram * count_feature))

        return score

    def compute_graph(self, pmi_vectors, k):
        matrix_length = 1000

        start = timeit.default_timer()
        dist = []
        for i in range(math.ceil(self.length / matrix_length)):
            print("%d  %d" % (i * matrix_length, self.length))
            if (i + 1) * matrix_length < self.length:
                v = pmi_vectors[i * matrix_length:(i + 1) * matrix_length, :]
            else:
                v = pmi_vectors[i * matrix_length:, :]
            # compute distance
            dist_vec = pairwise_distances(v, pmi_vectors, metric='cosine')
            dist.extend(dist_vec.argsort()[:, 1:k + 1])

        for i in range(self.length):
            self.graph_map[self.ngrams[i]] = [self.ngrams[j] for j in dist[i]]

        end = timeit.default_timer()
        print("consumed time: %s" % str(end - start))

    # raw ngram_list that contains duplicate itms
    def agg_marginal_prob(self, prob, ngram_list):
        ngram_index = 0
        for sent_prob in prob:
            for i in range(len(sent_prob) - 2):
                ngram = ngram_list[ngram_index]
                if ngram not in self.ngram_prob_map:
                    self.ngram_prob_map[ngram] = sent_prob[i + 1]
                else:
                    self.ngram_prob_map[ngram] = operate_dict(dict1=self.ngram_prob_map[ngram], dict2=sent_prob[i + 1],
                                                              operator='add')
                ngram_index += 1

        assert ngram_index == len(ngram_list)

        # test part
        # for ngram, prob in self.ngram_prob_map.items():
        #     number = ngram_counter[ngram]
        #     if number != 1:
        #         logger.debug("ngram numbers " + str(number))
        #         logger.debug("prob sum " + str(operate_dict(dict1=self.ngram_prob_map[ngram], operator='sum')))

    def propogate_graph(self):
        return

    # test function for debug graph
    def printGraph(self, print_num):
        count = 0

        for ngram, near in self.graph_map.items():
            near_set = []
            for index in near:
                near_set.append(self.ngrams[index])

            logger.debug(f'Graph: {ngram}\'s nearby: {"||".join(item for item in near_set)}')
            count += 1
            if count > print_num:
                break

