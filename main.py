from graph.helper import *
from graph.preprocess import preprocess_label, preprocess_unlabel
from graph.graph import *
from graph.pmi import *
import argparse
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import crf.crf_main as crf

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
    logger.info('**' * 20)
    logger.info('**' * 20)


class ModelData:
    # I/O
    word_emb_dir = None

    labeled_train_dir = None
    unlabeled_train_dir = None

    # data
    labeled_data = None
    unlabeled_data = None
    train_data = None

    # hyper parameters
    k_nearest = 3

    def __init__(self):
        pass

    def load_all_data(self):
        if self.labeled_train_dir:
            with open(self.labeled_train_dir) as f:
                raw_data = f.readlines()
            self.labeled_data = preprocess_label(raw_data)

        if self.unlabeled_train_dir:
            with open(self.unlabeled_train_dir) as f:
                raw_data = f.readlines()
            self.unlabeled_data = preprocess_unlabel(raw_data)

        self.train_data = self.labeled_data + self.unlabeled_data

    def get_graph_list(self):
        ngrams_list = []
        for sent in self.all_data:
            ngrams = sent2trigrams(sent)
            ngrams_list.extend(ngrams)

        features_list = []
        for sent in self.all_data:
            features = sent2graphfeatures(sent)
            features_list.extend(features)

        return ngrams_list, features_list

    def get_train_list(self, flag='POS'):
        text = []
        label = []
        if flag == 'POS':
            for sent in self.train_data:
                text_list = []
                label_list = []
                for word in sent:
                    text_list.append(normalize_word(word[0]))
                    label_list.append(word[2])
                text.append(text_list)
                label.append(label_list)
        elif flag == 'NER':
            for sent in self.train_data:
                text_list = []
                label_list = []
                for word in sent:
                    text_list.append(normalize_word(word[0]))
                    label_list.append(word[3])
                text.append(text_list)
                label.append(label_list)

        return text, label

    def build_word_emb(self):
        sentences, _ = self.get_train_list()

        model = Word2Vec(sentences, size=50)
        word_vector = model.wv
        model.save('data/word2vec.model')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_train", default='./data/labeled_train')
    parser.add_argument("--unlabeled_train", default='./data/unlabeled_train')
    args = parser.parse_args()

    # load config
    data = ModelData()
    data.labeled_train_dir = args.labeled_train
    data.unlabeled_train_dir = args.unlabeled_train

    # load dataset
    logger.debug("start reading data")
    data.load_all_data()
    logger.debug("length of label_data: " + str(len(data.labeled_data)))

    # build word embeddings
    data.build_word_emb()
    logger.debug("finish build word embeddings")

    logging_star()

    #
    # # create graph with labeled and unlabeled data
    # ngrams_list, graph_features_list = data.get_graph_list()
    # pmi = PMI(ngrams_list, graph_features_list)
    # pmi_vectors = pmi.pmi_vectors_improve()
    # logger.debug("finish construct vectors improve")
    #
    # # construct graph
    # graph = Graph(list(pmi.ngrams_feature_map.keys()), pmi_vectors, len(data.unlabeled_data), data.k_nearest)
    # logger.debug("finish Construct Graph")

    # posterior decoding

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


