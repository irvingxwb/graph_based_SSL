from collections import Counter
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics import pairwise_distances
import timeit
import logging
import math

logger = logging.getLogger("PMI")


class PMI:

    def __init__(self, ngrams_list, features_list):
        # self.features = set()
        self.feature_dict = {}

        keys = features_list[0].keys()
        feature_agg = dict.fromkeys(keys, [])
        ngrams_feature_agg = dict.fromkeys(keys, [])

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
                feature_agg[feature_name].append(feature)
                ngrams_feature_agg[feature_name].append((ngram, feature))

                if feature not in self.feature_dict:
                    self.feature_dict[feature] = feature_count
                    feature_count += 1

        # logging.debug(str(self.feature_dict))

        for feature_name, feature in feature_agg.items():
            # self.features.update(feature)
            self.feature_counters[feature_name] = Counter(feature)

        # self.features = list(self.features)
        self.sum_features = len(self.feature_dict)
        self.sum_ngrams = len(self.ngrams_feature_map)

        for feature_name, ngram_feature in ngrams_feature_agg.items():
            self.ngrams_feature_counters[feature_name] = Counter(ngram_feature)

        logger.debug("PMI: ngram_map size and feature_map size: " + str(len(self.ngrams_feature_map)) + " " + str(
            len(self.feature_dict)))
        logger.debug("PMI: complete pmi with: " + str(self.sum_ngrams) + " " + str(self.sum_features))

    def pmi_score(self, ngram, feature, feature_name):

        count_ngram_feature = self.ngrams_feature_counters[feature_name][(ngram, feature)]
        count_ngram = self.ngrams_counter[ngram]
        count_feature = self.feature_counters[feature_name][feature]

        score = np.log((count_ngram_feature * self.sum_ngrams) / (count_ngram * count_feature))

        return score

    # deprecated
    # def pmi_vectors_sparse(self):
    #     ngram_idx = 0
    #     ret_vectors = lil_matrix((self.sum_ngrams, self.sum_features))
    #     for ngram, features in self.ngrams_feature_map.items():
    #         for feature in features:
    #             for name, item in feature.items():
    #                 item_idx = self.features.index(item)
    #                 ret_vectors[ngram_idx, item_idx] = self.pmi_score(ngram, item, name)
    #
    #         ngram_idx += 1
    #
    #     return ret_vectors

    def pmi_vectors_improve(self):
        ngram_idx = 0
        ret_vectors = lil_matrix((self.sum_ngrams, self.sum_features))
        for ngram, features in self.ngrams_feature_map.items():
            for feature in features:
                for name, item in feature.items():
                    item_idx = self.feature_dict[item]
                    ret_vectors[ngram_idx, item_idx] = self.pmi_score(ngram, item, name)

            ngram_idx += 1

        return ret_vectors
