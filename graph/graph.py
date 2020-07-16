from collections import Counter
from .helper import operate_dict
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics import pairwise_distances
import timeit
import logging
import math

logger = logging.getLogger("Graph")


class Graph:
    # k for k_nearest
    def __init__(self, ngrams, pmi_vectors, unlabeled, k):
        self.ngrams = ngrams
        self.unlabeled = unlabeled
        self.graph_map = dict.fromkeys(ngrams)
        self.ngram_prob_map = {}
        self.length = len(ngrams)
        # compute k nearest map
        logger.debug(f'check length {str(len(self.ngrams))}  {str(pmi_vectors.shape[0])}')

        start = timeit.default_timer()
        self.compute_graph(pmi_vectors, k)
        end = timeit.default_timer()

        logger.debug(f'Compute graph complete: {str(end - start)}')
        # logger.debug(f'graph size {str(len(self.graph_map))}')

    def compute_graph(self, pmi_vectors, k):
        pmi_vectors = pmi_vectors.tocsr()
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

