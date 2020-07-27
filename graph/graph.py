from .functions import operate_dict, get_ngrams
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
    ngram_dict = {}
    feature_dict = {}

    ngrams_counter = None
    feature_counters = None
    ngrams_feature_counters = None
    ngrams_feature_map = None

    ngram_count = 0
    feature_count = 0
    # index of pmi vectors is the same as ngram dict
    pmi_vectors = None

    # graph
    graph_map = None
    ngram_prob_map = None

    # k for k_nearest
    def __init__(self, data_set):
        self.data_set = data_set
        self.build_feature_dicts()
        self.graph_map = dict.fromkeys(range(len(self.ngram_dict)))
        # compute k nearest map

    def __len__(self):
        return len(self.graph_map)

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
        self.feature_count = 0
        self.ngram_count = 0
        for ngram, features in zip(ngrams_list, features_list):
            if ngram not in self.ngrams_feature_map:
                self.ngrams_feature_map[ngram] = list()
                self.ngrams_feature_map[ngram].append(features)
            else:
                self.ngrams_feature_map[ngram].append(features)

            for feature_name, feature in features.items():
                feature_agg.append(feature)
                ngrams_feature_agg.append((ngram, feature))

                if feature not in self.feature_dict:
                    self.feature_dict[feature] = self.feature_count
                    self.feature_count += 1

            if ngram not in self.ngram_dict:
                self.ngram_dict[ngram] = self.ngram_count
                self.ngram_count += 1

        self.feature_counters = Counter(feature_agg)
        self.ngrams_feature_counters = Counter(ngrams_feature_agg)

        logger.debug("complete graph init with: %s %s" % (str(self.ngram_count), str(self.feature_count)))

    def build_pmi_vectors(self):

        self.pmi_vectors = torch.zeros(self.ngram_count, self.feature_count, dtype=torch.float)
        for n_idx, ngram in enumerate(self.ngram_dict):
            for features in self.ngrams_feature_map[ngram]:
                for feature_name, feature in features.items():
                    f_idx = self.feature_dict[feature]
                    score = self.pmi_score(ngram, feature)
                    self.pmi_vectors[n_idx, f_idx] = score

        logger.debug("complete pmi vectors compute")

    def pmi_score(self, ngram, feature):

        count_ngram_feature = self.ngrams_feature_counters[(ngram, feature)]
        count_ngram = self.ngrams_counter[ngram]
        count_feature = self.feature_counters[feature]

        score = np.log((count_ngram_feature * self.ngram_count) / (count_ngram * count_feature))

        return score

    def compute_graph(self):
        k = self.data_set.k_nearest
        # batch size to be 1000
        matrix_length = 1000
        total_length = len(self.ngram_dict)

        nearest_set = []
        sk_start = timeit.default_timer()
        for i in range(math.ceil(total_length / matrix_length)):
            if (i + 1) * matrix_length < total_length:
                v = self.pmi_vectors[i * matrix_length:(i + 1) * matrix_length, :]
            else:
                v = self.pmi_vectors[i * matrix_length:, :]
            # compute distance
            dist_vec = pairwise_distances(v, self.pmi_vectors, metric='cosine')
            batch_nearest_set = dist_vec.argsort()[:, 1:k + 1]
            nearest_set.extend(batch_nearest_set)
        sk_end = timeit.default_timer()
        sklearn_time = sk_end - sk_start

        assert len(nearest_set) == len(self.graph_map)
        for idx in range(len(nearest_set)):
            self.graph_map[idx] = nearest_set[idx]

        # deprecated because of low speed

        # cos = torch.nn.CosineSimilarity(dim=1)
        # vector_len = self.pmi_vectors.shape[1]
        # logger.debug("compute graph total length %s" % str(pmi_len))

        # torch_start = timeit.default_timer()
        # for idx in range(pmi_len):
        #     self_vector = self.pmi_vectors[idx]
        #     self_vector = self_vector.view(1, vector_len).expand(pmi_len, vector_len)
        #     distance = cos(self_vector, self.pmi_vectors)
        #     value, index = torch.topk(distance, k=k + 1)
        #     self.graph_map[idx] = [ele.item() for ele in index if ele.item() != idx]
        #     if idx == 100:
        #         logger.debug("finish compute to idx of %s" % str(idx))
        #         break
        # torch_end = timeit.default_timer()
        # torch_time = torch_end - torch_start

        logger.debug("complete building graph")

    # raw data with its crf probs
    def token2type_map(self, probs, mask, raw_data):
        for sent_probs, sent_mask, sent in zip(probs, mask, raw_data):
            if len(sent) < 3:
                pass
            else:
                ngrams = get_ngrams(sent)
                sent_probs = sent_probs[1:len(sent_probs)-1]
                assert len(ngrams) == ((sent_mask != 0).sum() - 2)
                for ngram, probs in zip(ngrams, sent_probs):
                    if ngram not in self.ngram_prob_map:
                        self.ngram_prob_map[ngram] = [probs]
                    else:
                        self.ngram_prob_map[ngram].append(probs)

                # get average probs and normalize
                for idx, ngram in enumerate(self.ngram_prob_map):
                    probs_agg = torch.cat(self.ngram_prob_map[ngram], dim=1)
                    probs_sum = torch.sum(probs_agg, dim=1)
                    probs_avg = probs_sum / probs_agg.shape[0]
                    self.ngram_prob_map[ngram] = probs_avg

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

    def get_flat_ngramfeature_lists(self):
        n_list = []
        f_list = []

        self.pmi_vectors = torch.zeros(self.ngram_count, self.feature_count, dtype=torch.float)
        for n_idx, ngram in enumerate(self.ngram_dict):
            n_list.append(ngram)
        for f_idx, feature in enumerate(self.feature_dict):
            f_list.append(feature)

        return n_list, f_list
