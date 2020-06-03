from collections import Counter
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics import pairwise_distances
import timeit
import logging
logger = logging.getLogger("Graph")


class Graph:
    # k for k_nearest
    def __init__(self, ngrams, pmi_vectors, unlabeled, k):
        self.ngrams = ngrams
        self.unlabeled = unlabeled
        self.graph_map = {}
        self.length = len(ngrams)
        # compute k nearest map
        logger.debug(f'check length {str(len(self.ngrams))}  {str(pmi_vectors.shape[0])}')

        start = timeit.default_timer()
        self.compute_graph(pmi_vectors, k)
        end = timeit.default_timer()

        logger.debug(f'Compute graph complete: {str(end-start)}')
        logger.debug(f'graph size {str(len(self.graph_map))}')

    def compute_graph(self, pmi_vectors, k):
        pmi_vectors = pmi_vectors.tocsr()

        for i in range(self.length):
            ngram = self.ngrams[i]
            v = pmi_vectors.getrow(i)
            dist_list = []

            # compute distance
            dist_vec = pairwise_distances(v, pmi_vectors, metric='cosine')[0]
            idx = dist_vec.argsort()[:k].tolist()
            self.graph_map[ngram] = [self.ngrams[i] for i in idx]
            logger.debug(f'{ngram} : {" ".join(self.graph_map[ngram])}')

    # test function for debug graph
    def printGraph(self, print_num):
        count = 0

        for ngram, near in self.graph_map.items():
            near_set = []
            for index in near:
                near_set.append(self.ngrams[index])

            logger.debug(f'Graph: {ngram}\'s nearby: {" ".join(item for item in near_set)}')
            count += 1
            if count > print_num:
                break


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

        logger.debug("PMI: ngram_map size and feature_map size: " + str(len(self.ngrams_feature_map)) +" "+ str(len(self.feature_dict)))
        logger.debug("PMI: complete pmi with: "+str(self.sum_ngrams)+" "+str(self.sum_features))

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


