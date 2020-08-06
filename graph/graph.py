from .functions import sent2trigrams, sent2graphfeatures
from sklearn.metrics import pairwise_distances
from collections import Counter
from graph.io_helper import *
import numpy as np
import logging
import torch
import copy

logger = logging.getLogger("Graph")


class Graph:
    # data
    train_text = None
    labeled_train_text = None
    unlabeled_train_text = None

    # train result
    tag_ngrams = None
    tag_seq = None
    tag_probs = None
    tag_mask = None

    # pmi dictionaries
    labeled_ngram_dict = dict()
    unlabeled_ngram_dict = dict()
    # dict for all ngrams = labeled + unlabeled
    # {instance: instance_idx}
    ngram_dict = dict()
    ngram_reverse_dict = dict()
    feature_dict = dict()

    # Counter objectives
    ngram_counters = None
    feature_counters = None
    ngrams_feature_counters = None
    ngrams_features_dict = dict()

    # number of each component
    ngram_count = 0
    feature_count = 0
    # index of pmi vectors is the same as ngram dict
    pmi_vectors = None

    # graph map is numpy nd array

    # [ngram_num, neighbour_num]
    graph_map = None
    # [ngram_num, neighbour_num]
    graph_weight_map = None

    # {ngram: probs}
    ngram_prob_map = dict()
    new_prob_map = dict()

    # parameter
    labeled_num = 0

    # k for k_nearest
    def __init__(self, data_set):
        self.data_set = data_set
        self.build_feature_dicts()
        # compute k nearest map

    def __len__(self):
        return len(self.graph_map)

    # save ngram_dict, graph_map, graph_weight_map
    def save(self, graph_dir, part):
        if part == 'graph':
            save_ins(self.ngram_dict, "ngram_dict", graph_dir)
            save_ins(self.graph_map, "graph_map", graph_dir)
            save_ins(self.graph_weight_map, "weight_map", graph_dir)
        elif part == 'propogate':
            pass

    def load(self, graph_dir, part):
        if part == 'graph':
            self.ngram_dict = load_ins("ngram_dict", graph_dir)
            self.graph_map = load_ins("graph_map", graph_dir)
            self.graph_weight_map = load_ins("weight_map", graph_dir)
        elif part == 'propogate':
            pass

    def update_train_result(self, tag_seq, tag_probs, tag_mask):
        self.tag_seq = tag_seq
        self.tag_probs = tag_probs
        self.tag_mask = tag_mask
        # assert tag_seq = max(tag_probs)

    def build_feature_dicts(self):
        # self.features = set()
        self.labeled_train_text, self.unlabeled_train_text, _ = self.data_set.get_train_text()
        self.train_text = self.labeled_train_text

        # agg
        ngram_sents_set = []
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
            ngram_sents_set.append(sent_ngrams)

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
            ngram_sents_set.append(sent_ngrams)

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

        # map each token in sents into trigram
        self.tag_ngrams = ngram_sents_set
        self.ngram_counters = Counter(ngram_agg)
        self.feature_counters = Counter(feature_agg)
        self.ngrams_feature_counters = Counter(ngrams_feature_agg)

        # build all ngrams dict
        self.ngram_dict = copy.deepcopy(self.labeled_ngram_dict)
        self.ngram_dict.update(self.unlabeled_ngram_dict)
        # build reverse ngram dict {idx : ngram}
        self.ngram_reverse_dict = {v: k for k, v in self.ngram_dict.items()}

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

    # compute neighbourhood
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
    def token2type_map(self, flag):
        logger.debug("token to type: flag %s" % flag)

        # init probs
        ngrams = self.tag_ngrams
        probs = self.tag_probs
        mask = self.tag_mask
        if flag == "train":
            sents = self.train_text

        self.ngram_prob_map = dict.fromkeys(self.ngram_dict.keys(), list())
        assert len(ngrams) == len(probs) == len(mask)
        for sent_ngrams, sent_probs, sent_mask in zip(ngrams, probs, mask):

            assert len(sent_ngrams) == ((sent_mask != 0).sum())
            # ngram is string of type [A B C] so that it can be key of the dict
            # num of ngrams is not the same as number of probs
            for n_idx, ngram in enumerate(sent_ngrams):
                # add n_probs
                n_probs = sent_probs[n_idx]
                if ngram not in self.ngram_prob_map:
                    print("key error for %s" % ngram)
                else:
                    self.ngram_prob_map[ngram].append(n_probs.view(1, 20))

        # get average probs for those types with more than one probs
        for idx, ngram in enumerate(self.ngram_prob_map):
            if len(self.ngram_prob_map[ngram]) == 1:
                pass
            else:
                probs_agg = torch.cat(self.ngram_prob_map[ngram], dim=0)
                probs_sum = torch.sum(probs_agg, dim=0)
                probs_avg = probs_sum / probs_agg.shape[0]
                self.ngram_prob_map[ngram] = probs_avg

        logger.debug("finish token to type map")

    # do graph propogations
    def graph_props(self, label_count, iter_num):
        # get empirical count for each label type
        logger.debug("start graph propogating with iteration number {iter_num} %d" % iter_num)
        count_r = self.ngramlist_and_sents2cr(label_count)
        r = self.build_r(count_r)
        mu = 0.5
        nu = 0.01
        self.new_prob_map = dict.fromkeys(self.ngram_dict.keys())
        # start propogate graph
        # probs = gamma(u) / kappa(u)

        # calculate parameters
        u_y = 1 / label_count
        for i in range(iter_num):
            # delta = 1 is ngram is labeled
            for n_idx, ngram in enumerate(self.ngram_prob_map):
                neighbour_sum = self.neighbour_sum(ngram)
                weight_sum = self.neighbour_weight_sum(ngram)
                gamma_u = self.delta(ngram) * r[ngram] + mu * neighbour_sum + nu * u_y
                kappa_u = self.delta(ngram) + nu + mu * weight_sum
                self.new_prob_map[ngram] = gamma_u / kappa_u

            # compare new and old for debugging
            match_cnt = 0
            change_cnt = 0
            none_o_cnt = 0
            none_o_change_cnt = 0
            for ngram, cnt in self.ngram_counters.items():
                a = self.ngram_prob_map[ngram]
                b = self.new_prob_map[ngram]
                _, a_idx = torch.max(a.view(1, -1), 1)
                _, b_idx = torch.max(b.view(1, -1), 1)
                if a_idx == b_idx:
                    match_cnt += 1
                    if a_idx != 2:
                        none_o_cnt += 1
                else:
                    change_cnt += 1
                    if a_idx != 2:
                        none_o_change_cnt += 1

            logger.debug("after one propogate, accuracy rate : %d" % (match_cnt / (match_cnt + change_cnt)))
            logger.debug("after one propogate, none O label accuracy rate : %d" % (none_o_cnt /
                                                                                   (none_o_cnt + none_o_change_cnt)))
            self.ngram_prob_map = self.new_prob_map
            logger.debug("finish propogate iteration : %d" % i)

        return self.new_prob_map

    # cr is a 2d torch tensor
    def ngramlist_and_sents2cr(self, label_count):
        cr = torch.zeros(self.ngram_count, label_count)
        tag_ngrams = self.tag_ngrams
        tag_seq = self.tag_seq
        ngram_type_counter = Counter()
        for sent_ngrams, sent_label in zip(tag_ngrams, tag_seq):
            for sent_idx, ngram in enumerate(sent_ngrams):
                ngram_idx = self.ngram_dict[ngram]
                label = sent_label[sent_idx].item()
                ngram_type_counter[label] += 1
                cr[ngram_idx][label] += 1
        return cr

    # build r to be dict for easier reference
    def build_r(self, count_r):
        r = dict.fromkeys(self.ngram_dict.keys())
        assert len(self.ngram_dict) == len(count_r)
        for ngram, labels_cnt in zip(self.ngram_dict, count_r):
            all_cnt = sum(labels_cnt)
            n_cnt = labels_cnt / all_cnt
            r[ngram] = n_cnt
        return r

    # get sum of neighbour nodes:  sum(w_uv * q(v, m-1))
    def neighbour_sum(self, u):
        u_idx = self.ngram_dict[u]
        u_neigh = self.graph_map[u_idx]
        sum_list = list()
        # node in neigh is a list of index
        # u_neigh : [v1, v2, v3, ]
        # u graoh_weight: [w1, w2, w3, ]
        for idx, v_idx in enumerate(u_neigh):
            v_ngram = self.ngram_reverse_dict[v_idx]
            v_probs = self.ngram_prob_map[v_ngram]
            v_weight = self.graph_weight_map[u_idx][idx]
            sum_list.append((v_probs * v_weight).view(1, -1))

        sum_tensor = torch.cat(sum_list, dim=0)
        return torch.sum(sum_tensor, dim=0).cpu()

    # get sum of neighbour nodes weights
    def neighbour_weight_sum(self, u):
        u_idx = self.ngram_dict[u]
        weight_list = self.graph_weight_map[u_idx]
        return sum(weight_list)

    # check if ngram is labeled or not
    def delta(self, ngram):
        if ngram in self.labeled_ngram_dict:
            return 1
        elif ngram in self.unlabeled_ngram_dict:
            return 0
        else:
            KeyError("ngram not in either labeled or unlabeled dict: %s" % ngram)

    # decode all marginals back to pos tags
    # mix marginal = alpha * crf_marginal + (1-alpha) * prop_result
    def viterbi_decode(self):
        alpha = 0.6
        decode_tag = []
        assert len(self.train_text) == len(self.tag_probs)
        for sent, sent_probs in zip(self.train_text, self.tag_probs):
            sent_tag = []
            for ngram, prob in sent_probs:
                pass

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

    def show_graph_sample(self):
        for ngram, count in self.ngram_counters.items():
            if count >= 5:
                n_idx = self.ngram_dict[ngram]
                n_neigh = self.graph_map[n_idx]
                n_neigh_ngram = [self.ngram_reverse_dict[v_idx] for v_idx in n_neigh]
                logger.debug("map type {%s} with neighbourhood: \n %s" % ngram, ' * '.join(n_neigh_ngram))

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
