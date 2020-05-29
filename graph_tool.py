from collections import Counter
from helper import string2number, sortVector
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.metrics import pairwise_distances
import logging


class Graph:
    # k for k_nearest
    def __init__(self, ngrams, pmi_vectors, unlabeled, k):
        self.ngrams = ngrams
        self.unlabeled = unlabeled
        self.graph_map = {}
        # compute k nearest map

        distance_matrix = pairwise_distances(pmi_vectors, metric="cosine")

        ngram_idx = 0
        for row in distance_matrix:
            print(row)
            self.graph_map[ngrams[ngram_idx]] = map(int, sorted(row)[1:k+1])
            ngram_idx += 1

        logging.debug("[Graph Debug]: graph size " + str(len(self.graph_map)))

    # test function for debug graph
    def printGraph(self, print_num):
        count = 0

        for ngram, near in self.graph_map.items():
            near_set = []
            for index in near:
                near_set.append(self.ngrams[index])

            logging.debug("[Graph Debug]" + '*'.join(ele for ele in near))
            logging.debug("[Graph Debug]" + ngram + " nearby: " + '**'.join(ele for ele in near_set))
            count += 1
            if count > print_num:
                break


class PMI:
    def __init__(self, ngrams_list, features_list):
        self.features = set()

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
        for ngram, feature in zip(ngrams_list, features_list):
            if ngram not in self.ngrams_feature_map:
                self.ngrams_feature_map[ngram] = [feature]
            else:
                self.ngrams_feature_map[ngram].append(feature)

            for feature_name, feature_item in feature.items():
                feature_agg[feature_name].append(feature_item)
                ngrams_feature_agg[feature_name].append((ngram, feature_item))

        for feature_name, feature in feature_agg.items():
            self.features.update(feature)
            self.feature_counters[feature_name] = Counter(feature)

        self.features = list(self.features)
        self.sum_features = len(self.features)
        self.sum_ngrams = len(self.ngrams_feature_map)

        for feature_name, ngram_feature in ngrams_feature_agg.items():
            self.ngrams_feature_counters[feature_name] = Counter(ngram_feature)

        # for ngram in self.ngrams_feature_map:
        #     assert len(self.ngrams_feature_map[ngram])==self.ngrams_counter[ngram]

        logging.debug("complete pmi with: "+str(self.sum_ngrams)+" "+str(self.sum_features))

    def pmi_score(self, ngram, feature, feature_name):
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

    def pmi_vectors_hash(self):
        ret_vecotrs = []
        for ngram, features in self.ngrams_feature_map.items():
            ngram_vector = []
            for feature in features:
                for name, item in feature.items():
                    ngram_vector.append([self.pmi_score(ngram, item, name), string2number(item)])

            # sort the vector for simplify similarity computation
            ngram_vector = sortVector(ngram_vector)
            ret_vecotrs.append(ngram_vector)

        return ret_vecotrs

    def pmi_vectors_sparse(self):
        ngram_idx = 0
        ret_vectors = lil_matrix((self.sum_ngrams, self.sum_features))
        for ngram, features in self.ngrams_feature_map.items():
            for feature in features:
                for name, item in feature.items():
                    item_idx = self.features.index(item)
                    ret_vectors[ngram_idx, item_idx] = self.pmi_score(ngram, item, name)

            ngram_idx += 1

        return ret_vectors


