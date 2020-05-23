from collections import Counter
import numpy as np


class Graph:
    # k for k_nearest
    def __init__(self, ngrams, pmi_vectors, unlabeled, k):
        self.ngrams = ngrams
        self.unlabeled = unlabeled
        self.graph_dic = {}


class PMI:
    def __init__(self, ngrams, features_list, features_flatten_list):
        self.sum_ngrams = len(ngrams)
        self.features_flatten_list = features_flatten_list
        self.vector_len = len(features_flatten_list)

        keys = features_list[0].keys()
        feature_agg = dict.fromkeys(keys, [])
        ngram_feature_agg = dict.fromkeys(keys, [])

        # gather all features by its name
        for ngram, feature in zip(ngrams, features_list):
            for feature_name, feature_item in feature.items():
                feature_agg[feature_name].append(feature_item)
                ngram_feature_agg[feature_name].append((ngram, feature_item))

        self.ngrams_counter = Counter(ngrams)
        self.feature_counters = {}
        self.ngrams_feature_counters = {}

        for feature_name, feature in feature_agg.items():
            self.feature_counters[feature_name] = Counter(feature)

        for feature_name, ngram_feature in ngram_feature_agg.items():
            self.ngrams_feature_counters[feature_name] = Counter(ngram_feature)

    def pmi(self, ngram, feature, feature_name):
        if self.ngrams_feature_counters[feature_name][(ngram, feature)] is not 0:
            count_ngram_feature = self.ngrams_feature_counters[feature_name][(ngram, feature)]
        else:
            return 0

        if self.ngrams_counter[ngram] is not 0:
            count_ngram = self.ngrams_counter[ngram]
        else:
            return 0

        if self.feature_counters[feature_name][feature] is not 0:
            count_feature = self.feature_counters[feature_name][feature]
        else:
            return 0

        score = np.log((count_ngram_feature * self.sum_ngrams) / (count_ngram * count_feature))

        return score

    def pmi_vector(self, ngram, feature):
        ret_arr = np.zeros(self.vector_len)

        for feature_name, item in feature.items():
            feature_idx = self.features_flatten_list.index(item)
            ret_arr[feature_idx] = self.pmi(ngram, item, feature_name)

        return ret_arr
        #
        # return np.array(
        #     [self.pmi(ngram, feature, feature_name) for feature_name, feature_set in features_dict.items() for feature
        #      in feature_set])
