from graph.io_helper import *
from graph.functions import *
from graph.graph import *
import argparse
import torch
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from crf.crf_main import NCRFpp
from crf.utils.functions import normalize_word

import timeit
from sys import stdout
import logging

# logging.basicConfig(level=logging.DEBUG, filename='./log/main.log', filemode='w', format='%(asctime)s:%(levelname)s:%(
# name)s:%(message)s')
logging.basicConfig(level=logging.DEBUG, stream=stdout, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logger = logging.getLogger('Main')


def normalize_word(word):
    if word.isdigit():
        return '_NUMBER'
    else:
        return word


def logging_star():
    logger.info('**' * 40)
    logger.info('**' * 40)


def preprocess_data(raw_data):
    text = []
    label = []
    text_sent = []
    label_sent = []
    for line in raw_data:
        if line == '\n' and len(text_sent) != 0:
            text.append(text_sent)
            label.append(label_sent)
            text_sent = []
            label_sent = []
        else:
            sent = line.split()
            text_sent.append(sent[0])
            label_sent.append(sent[1])

    if len(text_sent) != 0:
        text.append(text_sent)
        label.append(label_sent)

    return text, label


# normalize probabilities
def normal_probs(probs):
    new_probs = []
    for sent_probs in probs:
        new_sent_probs = []
        for type_prob in sent_probs:
            # normalize tensor to [-1, 1]
            interval = torch.max(type_prob) - torch.min(type_prob)
            type_prob = type_prob / interval
            # get probabilities using Softmax
            type_prob = torch.nn.Softmax(type_prob).dim
            new_sent_probs.append(type_prob)

        new_probs.append(new_sent_probs)

    logger.debug("normalized probabilities")
    return new_probs


def test_decode_result(tag_seq, tag_probs, tag_mask):
    assert len(tag_seq) == len(tag_probs)
    for seq, probs in zip(tag_seq, tag_probs):
        assert len(seq) == len(probs)
        temp, idx = torch.max(probs, 1)
        print(1)


# function for save and load crf result
def save_crf_result(tag_seq, tag_probs, tag_mask, file_dir):
    save_tensor(tag_seq, "tag_seq", file_dir)
    save_tensor(tag_probs, "tag_probs", file_dir)
    save_tensor(tag_mask, "tag_mask", file_dir)


def load_crf_result(file_dir):
    tag_seq = load_tensor("tag_seq", file_dir)
    tag_probs = load_tensor("tag_probs", file_dir)
    tag_mask = load_tensor("tag_mask", file_dir)
    return tag_seq, tag_probs, tag_mask


class Dataset:
    # I/O
    word_emb_dir = None
    labeled_train_dir = None
    unlabeled_train_dir = None

    # data
    labeled_train_text = None
    labeled_train_label = None
    unlabeled_train_text = None
    train_data = None
    all_data = None
    crf_data = None

    # hyper parameters
    k_nearest = 5
    unlabeled_num = 0
    gpu = False

    def __init__(self):
        pass

    def load_all_data(self):
        if self.labeled_train_dir:
            with open(self.labeled_train_dir) as f:
                raw_data = f.readlines()
            self.labeled_train_text, self.labeled_train_label = preprocess_data(raw_data)

        self.all_data = self.labeled_train_text
        self.unlabeled_train_text = []

        if self.unlabeled_train_dir:
            with open(self.unlabeled_train_dir) as f:
                raw_data = f.readlines()
            self.unlabeled_train_text= preprocess_data(raw_data)

    def get_features_list(self):
        features_list = []
        for sent in self.labeled_train_text:
            features = sent2graphfeatures(sent)
            features_list.extend(features)

        return features_list

    def get_train_text(self, flag='POS'):
        return self.labeled_train_text, self.unlabeled_train_text, self.labeled_train_label

    # todo: produce word emb instead of using pre-trained
    def build_word_emb(self):
        sentences, _, _ = self.get_train_text()

        model = Word2Vec(sentences, size=50)
        word_vector = model.wv
        model.save('data/word2vec.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_train", default='./data/train.bmes')
    parser.add_argument("--unlabeled_train", default='./data/train.bme')
    parser.add_argument("--graph_dir", default='./data/save/graph/')
    parser.add_argument("--crf_dir", default='./data/save/crf/')
    args = parser.parse_args()

    # debug mode use all prestored data for debugging
    load_graph = True
    load_crf = True

    # add crf data structure to main data set
    data_set = Dataset()
    data_set.gpu = torch.cuda.is_available()
    data_set.labeled_train_dir = args.labeled_train
    data_set.unlabeled_train_dir = args.unlabeled_train

    # load data set
    data_set.load_all_data()
    logger.debug("length of label_data: " + str(len(data_set.labeled_train_text)))

    # initialize graph
    graph = Graph(data_set)

    if not load_graph:
        graph.build_pmi_vectors()
        graph.construct_graph()
        graph.save(args.graph_dir, 'graph')
    else:
        graph.load(args.graph_dir, 'graph')
        logger.debug("Load pre-computed graph from file")

    # initialize crf
    if not load_crf:
        # init crf, time consuming
        crf = NCRFpp(data_set)
        crf.build_crf()
        tag_seq, tag_probs, tag_mask = crf.decode_marginals("train")
        save_crf_result(tag_seq, tag_probs, tag_mask, args.crf_dir)
        logger.debug("finish crf train")
    else:
        tag_seq, tag_probs, tag_mask = load_crf_result(args.crf_dir)
        graph.update_train_result(tag_seq, tag_probs, tag_mask)

        logger.debug("load crf result from file")

    graph.update_train_result(tag_seq, tag_probs, tag_mask)

    # token to type map
    graph.token2type_map(flag='train')

    # graph propogations
    graph.graph_props(20, 10)

    # Viterbi decoding
    graph.viterbi_decode()

    # Retrain crf