from .functions import sent2trigrams, sent2graphfeatures
from sklearn.metrics import pairwise_distances
from collections import Counter
import numpy as np
import logging
import torch
import copy

logger = logging.getLogger("Graph")


class Graph:
    # data
    crf_data = None
    labeled_train_text = None
    unlabeled_train_text = None
    test_text = None

    # pmi dictionaries
    labeled_ngram_dict = dict()
    unlabeled_ngram_dict = dict()
    # dict for all ngrams = labeled + unlabeled
    ngram_dict = dict()
    feature_dict = dict()

    ngram_counters = None
    feature_counters = None
    ngrams_feature_counters = None
    ngrams_features_dict = dict()

    ngram_count = 0
    feature_count = 0
    # index of pmi vectors is the same as ngram dict
    pmi_vectors = None

    # graph map is numpy nd array
    graph_map = None
    graph_weight_map = None
    ngram_prob_map = dict()
    new_prob_map = dict()

    # parameter
    labeled_num = 0

    # k for k_nearest
    def __init__(self, data_set, crf):
        self.data_set = data_set
        self.crf_data = crf.data
        self.build_feature_dicts()
        # compute k nearest map

    def __len__(self):
        return len(self.graph_map)

    def build_feature_dicts(self):
        # self.features = set()
        self.labeled_train_text, self.unlabeled_train_text, _ = self.data_set.get_train_text()

        # agg
        ngram_agg = []
        feature_agg = []
        ngrams_feature_agg = []

        # gather all features by its name
        self.feature_count = 0
        self.ngram_count = 0

        # manipulate labeled data
        for sent in self.labeled_train_text:
            # get ngrams, features from sentence
            sent_ngrams = sent2trigrams(sent)
            sent_features = sent2graphfeatures(sent)
            # build maps and dicts
            assert len(sent_ngrams) == len(sent_features)
            for ngram, features in zip(sent_ngrams, sent_features):
                # manipulate labeled ngrams
                if ngram not in self.labeled_ngram_dict.keys():
                    self.labeled_ngram_dict[ngram] = self.ngram_count
                    self.ngram_count += 1
                    self.ngrams_features_dict[ngram] = [features]
                    ngram_agg.append(ngram)
                else:
                    self.ngrams_features_dict[ngram].append(features)
                    ngram_agg.append(ngram)

                # manipulate features
                for _, feature in features.items():
                    if feature not in self.feature_dict:
                        self.feature_dict[feature] = self.feature_count
                        self.feature_count += 1
                    feature_agg.append(feature)
                    ngrams_feature_agg.append((ngram, feature))

        # handle unlabeled data
        for sent in self.unlabeled_train_text:
            # get ngrams, features from sentence
            sent_ngrams = sent2trigrams(sent)
            sent_features = sent2graphfeatures(sent)
            # build maps and dicts
            for ngram, probs in zip(sent_ngrams, sent_features):
                # manipulate labeled ngrams
                if ngram not in self.labeled_ngram_dict.keys():
                    self.unlabeled_ngram_dict[ngram] = self.ngram_count
                    self.ngram_count += 1
                    self.ngrams_features_dict[ngram] = [features]
                    ngram_agg.append(ngram)
                else:
                    self.ngrams_features_dict[ngram].append(features)
                    ngram_agg.append(ngram)

                # manipulate features
                for _, feature in features.items():
                    if feature not in self.feature_dict:
                        self.feature_dict[feature] = self.feature_count
                        self.feature_count += 1
                    feature_agg.append(feature)
                    ngrams_feature_agg.append((ngram, feature))

        self.ngram_counters = Counter(ngram_agg)
        self.feature_counters = Counter(feature_agg)
        self.ngrams_feature_counters = Counter(ngrams_feature_agg)

        # build all ngrams dict
        self.ngram_dict = copy.deepcopy(self.labeled_ngram_dict)
        self.ngram_dict.update(self.unlabeled_ngram_dict)

        logger.debug("complete graph init with: %s %s" % (str(self.ngram_count), str(self.feature_count)))

    def build_pmi_vectors(self):
        self.pmi_vectors = torch.zeros(self.ngram_count, self.feature_count, dtype=torch.float)
        for n_idx, ngram in enumerate(self.ngram_dict):
            for features in self.ngrams_features_dict[ngram]:
                for feature_name, feature in features.items():
                    f_idx = self.feature_dict[feature]
                    score = self.pmi_score(ngram, feature)
                    self.pmi_vectors[n_idx, f_idx] = score

        logger.debug("complete pmi vectors compute")

    def pmi_score(self, ngram, feature):
        count_ngram = self.ngram_counters[ngram]
        count_feature = self.feature_counters[feature]
        count_ngram_feature = self.ngrams_feature_counters[(ngram, feature)]

        score = np.log((count_ngram_feature * self.ngram_count) / (count_ngram * count_feature))

        return score

    def construct_graph(self):
        k = self.data_set.k_nearest

        dist_vec = pairwise_distances(self.pmi_vectors, self.pmi_vectors, metric='cosine')
        nearest_set = dist_vec.argsort()[:, 1:k + 1]

        logger.debug("finish computer distance matrix")
        # each node is a list of neighbour nodes
        self.graph_map = nearest_set

        # construct neighbour nodes' matching weight
        self.graph_weight_map = []
        for u, u_neigh in enumerate(self.graph_map):
            # check if u is in K(v) for each v in K(u)
            u_weight = []
            for v_idx, v in enumerate(u_neigh):
                if u in self.graph_map[v]:
                    assert dist_vec[u, v] == dist_vec[v, u]
                    u_weight.append(dist_vec[u, v])
                else:
                    u_weight.append(0)
            self.graph_weight_map.append(u_weight)

        logger.debug("complete building graph")

    # raw data with its crf probs
    def token2type_map(self, probs, mask, flag):
        logger.debug("token to type total sentences length %s" % flag)

        # init probs
        sents = None
        if flag == "train":
            sents = self.labeled_train_text
        elif flag == "text":
            sents = self.test_text

        assert len(sents) == len(probs) == len(mask)
        for sent, sent_probs, sent_mask in zip(sents, probs, mask):
            ngrams = sent2trigrams(sent)

            assert len(ngrams) == ((sent_mask != 0).sum())
            # ngram is string of type [A B C] so that it can be key of the dict
            for n_idx, ngram in enumerate(ngrams):
                n_probs = sent_probs[n_idx]
                if ngram not in self.ngram_dict:
                    print("key error for %s" % ngram)

                # add n_probs
                if ngram not in self.ngram_prob_map:
                    self.ngram_prob_map[ngram] = list()
                    self.ngram_prob_map[ngram].append(n_probs.view(1, 20))
                else:
                    self.ngram_prob_map[ngram].append(n_probs.view(1, 20))

        # get average probs and normalize
        for idx, ngram in enumerate(self.ngram_prob_map):
            if len(self.ngram_prob_map[ngram]) == 1:
                pass
            else:
                probs_agg = torch.cat(self.ngram_prob_map[ngram], dim=0)
                probs_sum = torch.sum(probs_agg, dim=0)
                probs_avg = probs_sum / probs_agg.shape[0]
                self.ngram_prob_map[ngram] = probs_avg

    # do graph propogations
    def graph_props(self, tag_text, tag_seq, label_dict):
        # get empirical count for each label type
        count_r = self.ngramlist_and_sents2cr(tag_text, tag_seq, label_dict=label_dict)
        r = self.build_r(count_r)
        mu = 0.5
        nu = 0.1
        self.new_prob_map = dict.fromkeys(self.ngram_dict.keys())
        # start propogate graph
        # probs = gamma(u) / kappa(u)\

        # calculate parameters
        u_y = 1 / len(label_dict)

        # delta = 1 is ngram is labeled
        for idx, ngram in self.ngram_prob_map:
            gamma_u = self.delta(ngram) * r[ngram] + mu * self.neighbour_sum(ngram) + nu * u_y
            kappa_u = self.delta(ngram) + nu + mu * self.neighbour_weight_sum(ngram)
            self.new_prob_map[ngram] = gamma_u / kappa_u

    @staticmethod
    def ngramlist_and_sents2cr(tag_text, tag_seq, label_dict):
        cr = {}
        label_keys_length = len(label_dict)
        ngram_type_counter = Counter()
        for sent, sent_label in zip(tag_text, tag_seq):
            ngrams = sent2trigrams(sent)
            assert len(ngrams) == len(sent_label)

            for ngram, label in zip(ngrams, sent_label):
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
        return 0

    # get sum of neighbour nodes weights
    def neighbour_weight_sum(self, ngram):
        return 0

    # check if ngram is labeled or not
    def delta(self, ngram):
        if ngram in self.labeled_ngram_dict:
            return 1
        elif ngram in self.unlabeled_ngram_dict:
            return 0
        else:
            KeyError("ngram not in either labeled or unlabeled dict: %s" % ngram)

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

# deprecated code

## construct graph part
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
