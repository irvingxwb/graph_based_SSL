from .functions import sent2trigrams
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
    crf_data = None

    # pmi
    ngram_dict = {}
    feature_dict = {}
    isngram_labeled_dict = {}

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
    graph_weight_map = None
    ngram_prob_map = None
    new_prob_map = None

    # k for k_nearest
    def __init__(self, data_set, crf):
        self.data_set = data_set
        self.crf_data = crf.data
        self.build_feature_dicts()
        self.graph_map = dict.fromkeys(range(len(self.ngram_dict)))
        # compute k nearest map

    def __len__(self):
        return len(self.graph_map)

    def build_feature_dicts(self):
        # self.features = set()
        labeled_ngrams_list, unlabeled_ngrams_list = self.data_set.get_ngrams_list()
        features_list = self.data_set.get_features_list()

        # temporary
        all_ngrams_list = labeled_ngrams_list

        feature_agg = []
        ngrams_feature_agg = []

        self.ngrams_counter = Counter(all_ngrams_list)
        # count feature numbers
        self.feature_counters = {}
        # count (ngram, feature) numbers
        self.ngrams_feature_counters = {}
        # each ngram got different featurs
        self.ngrams_feature_map = {}

        # gather all features by its name
        self.feature_count = 0
        self.ngram_count = 0

        # divide whether ngrams is labeled or not
        for ngram in labeled_ngrams_list:
            self.isngram_labeled_dict[ngram] = 1

        for ngram in unlabeled_ngrams_list:
            self.isngram_labeled_dict[ngram] = 0

        for ngram, features in zip(all_ngrams_list, features_list):
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

    def construct_graph(self):
        k = self.data_set.k_nearest

        # deprecated
        # # batch size to be 1000
        # matrix_length = 1000
        # total_length = len(self.ngram_dict)

        # for i in range(math.ceil(total_length / matrix_length)):
        #     if (i + 1) * matrix_length < total_length:
        #         v = self.pmi_vectors[i * matrix_length:(i + 1) * matrix_length, :]
        #     else:
        #         v = self.pmi_vectors[i * matrix_length:, :]
        #     # compute distance
        #     dist_vec = pairwise_distances(v, self.pmi_vectors, metric='cosine')
        #     batch_nearest_set = dist_vec.argsort()[:, 1:k + 1]
        #     nearest_set.extend(batch_nearest_set)
        nearest_set = []

        dist_vec = pairwise_distances(self.pmi_vectors, self.pmi_vectors, metric='cosine')
        batch_nearest_set = dist_vec.argsort()[:, 1:k + 1]
        nearest_set.extend(batch_nearest_set)

        assert len(nearest_set) == len(self.graph_map)
        # each node is a list of neighbour nodes
        self.graph_map = nearest_set

        # construct neighbour nodes' matching weight
        self.graph_weight_map = self.graph_map
        for u, u_neigh in enumerate(self.graph_map):
            # check if u is in K(v) for each v in K(u)
            u_weight = []
            for v_idx, v in enumerate(u_neigh):
                if u in self.graph_map[v]:
                    pass
                else:
                    u_weight[v_idx]

        logger.debug("complete building graph")

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

    # raw data with its crf probs
    def token2type_map(self, probs, mask, raw_data):
        logger.debug("token to type total sentences length %s" % str(len(raw_data)))

        # init probs
        self.ngram_prob_map = dict.fromkeys(self.ngram_dict.keys(), [])

        for sent_probs, sent_mask, sent in zip(probs, mask, raw_data):
            ngrams = sent2trigrams(sent)
            sent_probs = sent_probs[0:len(sent_probs)]
            assert len(ngrams) == ((sent_mask != 0).sum())
            # ngram is tuple so that it can be key of the dict
            for ngram, probs in zip(ngrams, sent_probs):
                if ngram not in self.ngram_prob_map:
                    print("key error for %s" % ngram)
                else:
                    self.ngram_prob_map[ngram].append(probs.view(1, 20))

        # get average probs and normalize
        for idx, ngram in enumerate(self.ngram_prob_map):
            if len(self.ngram_prob_map[ngram]) == 1:
                pass
            else:
                probs_agg = torch.cat(self.ngram_prob_map[ngram], dim=0)
                probs_sum = torch.sum(probs_agg, dim=0)
                probs_avg = probs_sum / probs_agg.shape[0]
                self.ngram_prob_map[ngram] = probs_avg

        # test part
        # for ngram, prob in self.ngram_prob_map.items():
        #     number = ngram_counter[ngram]
        #     if number != 1:
        #         logger.debug("ngram numbers " + str(number))
        #         logger.debug("prob sum " + str(operate_dict(dict1=self.ngram_prob_map[ngram], operator='sum')))

    def ngramlist_and_sents2cr(self, tag_text, tag_seq, label_dict):
        cr = {}
        label_keys_length = len(label_dict)
        ngram_type_counter = Counter()
        for sent, sent_label in zip(tag_text, tag_seq):
            ngrams = sent2trigrams(sent)
            assert len(ngrams) == len(sent_label)

            for ngram, label in zip(ngram, sent_label):
                ngram_type_counter[label] += 1
                if ngram not in cr:
                    cr[ngram] = dict.fromkeys(range(label_keys_length))
                cr[ngram][label] += 1

        return cr

    @staticmethod
    def build_r(count_r):
        r = {}
        for ngram, label_cnt in count_r.items():
            all_cnt = sum([cnt for cnt in label_cnt.values()])
            r[ngram] = [(cnt / all_cnt) for label, cnt in label_cnt.items()]

        return r

    # get sum of neighbour nodes:  sum(w_uv * q(v, m-1))
    def neighbour_sum(self, ngram):
        return

    # get sum of neighbour nodes weights
    def neighbour_weight_sum(self, ngram):
        return

    # check if ngram is labeled or not
    def delta(self, ngram):
        return self.isngram_labeled_dict[ngram]

    # do graph propogations
    def graph_props(self, tag_text, tag_seq, label_dict):
        # get empirical count for each label type
        count_r = self.ngramlist_and_sents2cr(tag_text, tag_seq, label_dict=label_dict)
        r = self.build_r(count_r)
        mu = 0.5
        nu = 0.1
        self.new_prob_map = dict.fromkeys(self.ngram_dict.keys())
        # start propogate graph
        # propogated probs = gamma(u) / kappa(u)\

        # calculate parameters
        u_y = 1 / len(label_dict)

        # delta = 1 is ngram is labeled
        for idx, ngram in self.ngram_prob_map:
            gamma_u = self.delta(ngram) * r[ngram] + mu * self.neighbour_sum(ngram) + nu * u_y
            kappa_u = self.delta(ngram) + nu + mu * self.neighbour_weight_sum(ngram)
            self.new_prob_map[ngram] = gamma_u / kappa_u

    def viterbi_decode(self):
        return

    def generate_retrain_data(self):
        return

    def get_flat_ngramfeature_lists(self):
        n_list = []
        f_list = []

        self.pmi_vectors = torch.zeros(self.ngram_count, self.feature_count, dtype=torch.float)
        for n_idx, ngram in enumerate(self.ngram_dict):
            n_list.append(ngram)
        for f_idx, feature in enumerate(self.feature_dict):
            f_list.append(feature)

        return n_list, f_list


