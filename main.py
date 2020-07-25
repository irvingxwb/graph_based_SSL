from graph.functions import *
from graph.graph import *
import argparse
import torch
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from crf.crf_main import NCRFpp

import timeit
from sys import getsizeof, stdout
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


class Dataset:
    # I/O
    word_emb_dir = None
    labeled_train_dir = None
    unlabeled_train_dir = None

    # data
    labeled_train_text = None
    labeled_train_label = None
    train_data = None
    all_data = None

    # hyper parameters
    k_nearest = 3
    unlabeled_num = 0
    gpu = False

    def __init__(self):
        pass

    def preprocess_data(self, raw_data):
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

        return text, label

    def load_all_data(self):
        if self.labeled_train_dir:
            with open(self.labeled_train_dir) as f:
                raw_data = f.readlines()
            self.labeled_train_text, self.labeled_train_label = self.preprocess_data(raw_data)

        self.all_data = self.labeled_train_text

        # if self.unlabeled_train_dir:
        #     with open(self.unlabeled_train_dir) as f:
        #         raw_data = f.readlines()
        #     self.unlabeled_trai= preprocess_data(raw_data)

    def get_graph_list(self):
        ngrams_list = []
        for sent in self.labeled_train_text:
            ngrams = sent2trigrams(sent)
            ngrams_list.extend(ngrams)

        features_list = []
        for sent in self.labeled_train_text:
            features = sent2graphfeatures(sent)
            features_list.extend(features)

        return ngrams_list, features_list

    def get_train_list(self, flag='POS'):
        return self.labeled_train_text, self.labeled_train_label

    def build_word_emb(self):
        sentences, _ = self.get_train_list()

        model = Word2Vec(sentences, size=50)
        word_vector = model.wv
        model.save('data/word2vec.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_train", default='./data/train.bmes')
    parser.add_argument("--unlabeled_train", default=None)
    args = parser.parse_args()

    # load config
    data_set = Dataset()
    data_set.gpu = torch.cuda.is_available()
    data_set.labeled_train_dir = args.labeled_train
    data_set.unlabeled_train_dir = args.unlabeled_train

    # load data set
    logger.debug("start reading data")
    data_set.load_all_data()
    logger.debug("length of label_data: " + str(len(data_set.labeled_train_text)))

    # # build word embeddings
    # data_set.build_word_emb()
    # logger.debug("finish build word embeddings")

    # initialize graph
    graph = Graph(data_set)
    graph.build_pmi_vectors()
    logger.debug("finish Construct Graph")

    # posterior decoding
    logging_star()

    crf = NCRFpp()
    crf.build_alphabet()
    predict_tag, predict_probs, acc = crf.decode_marginals()
    for tag, prob in zip(predict_tag, predict_probs):
        a, b = prob.max(dim=1)
    logger.debug("decode accuracy %s" % str(acc))
    logger.debug("finish crf train")

    #
    # # token to type map
    # graph.agg_marginal_prob(marginal_prob, ngrams_list)
    #
    # # graph propogation
    # graph.propogate_graph()

    # Viterbi decoding

    # start = timeit.default_timer()
    # graph.propogate_graph()
    # end = timeit.default_timer()
    # logger.debug("propograte graph: " + str(end - start))
    print("program finished")


