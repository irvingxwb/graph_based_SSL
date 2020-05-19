from collections import Counter
from sklearn.metrics import pairwise_distances
import numpy as np


class Graph:
    # k for k_nearest
    def __init__(self, ngrams, pmi_vectors, unlabeled, k):
        self.ngrams = ngrams
        self.sim_matrix = pairwise_distances(pmi_vectors, metric='cosine')
        self.unlabeled = unlabeled
        self.graph_dic = {}

        for i in range(self.sim_matrix.shape[0]):
            arr = np.argsort(self.sim_matrix[0], axis=1)
            self.graph_dic[ngrams[i]] = [arr[-idx-1] for idx in range(k)]


class PMI:
    def __init__(self, ngrams, features_list):
        self.ngrams_counter = Counter(ngrams)
        self.sum_ngrams = len(ngrams)

        feature_dict = {}
        ngrams_feature_dict = {}

        for ngram, features in zip(ngrams, features_list):
            for feature_name, feature in features.items():
                if feature_name not in feature_dict:
                    feature_dict[feature_name] = []
                if feature_name not in ngrams_feature_dict:
                    ngrams_feature_dict[feature_name] = []

                # dict for all context features
                feature_dict[feature_name].append(feature)
                # dict for all context features with (ngram, feature) pair as elements
                ngrams_feature_dict[feature_name].append((ngram, feature))

        self.feature_counters = {}
        self.ngrams_feature_counters = {}

        for feature_name, feature in feature_dict.items():
            self.feature_counters[feature_name] = Counter(feature)

        for feature_name, ngram_feature in ngrams_feature_dict.items():
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

    def pmi_vector(self, ngram, features_dict):
        return np.array(
            [self.pmi(ngram, feature, feature_name) for feature_name, feature_set in features_dict.items() for feature
             in feature_set])
