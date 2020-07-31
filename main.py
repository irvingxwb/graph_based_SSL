from graph.functions import *
from graph.graph import *
import argparse
import torch
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from crf.crf_main import NCRFpp
from crf.utils.functions import normalize_word

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

        # if self.unlabeled_train_dir:
        #     with open(self.unlabeled_train_dir) as f:
        #         raw_data = f.readlines()
        #     self.unlabeled_trai= preprocess_data(raw_data)

    def get_features_list(self):
        features_list = []
        for sent in self.labeled_train_text:
            features = sent2graphfeatures(sent)
            features_list.extend(features)

        return features_list

    def get_train_text(self, flag='POS'):
        return self.labeled_train_text, self.unlabeled_train_text, self.labeled_train_label

    def build_word_emb(self):
        sentences, _, _ = self.get_train_text()

        model = Word2Vec(sentences, size=50)
        word_vector = model.wv
        model.save('data/word2vec.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_train", default='./data/train.bmes')
    parser.add_argument("--unlabeled_train", default=None)
    args = parser.parse_args()

    # add crf data structure to main data set
    data_set = Dataset()
    data_set.gpu = torch.cuda.is_available()
    data_set.labeled_train_dir = args.labeled_train
    data_set.unlabeled_train_dir = args.unlabeled_train

    # load data set
    data_set.load_all_data()
    logger.debug("length of label_data: " + str(len(data_set.labeled_train_text)))

    # init crf
    crf = NCRFpp()
    crf.build_crf()

    # initialize all class
    graph = Graph(data_set, crf=crf)
    graph.build_pmi_vectors()
    graph.construct_graph()

    # posterior decoding
    tag_seq, tag_probs, tag_mask = crf.decode_marginals("train")
    tag_probs = normal_probs(tag_probs)
    # instance_Ids = [instance[0] for instance in crf.data.train_Ids]
    # instance_words = [[crf.data.word_alphabet.get_instance(word) for word in sent] for sent in instance_Ids]
    logger.debug("finish crf train")

    # token to type map
    graph.token2type_map(tag_probs, tag_mask, flag="train")

    # graph propogations
    graph.graph_props(data_set.labeled_train_text, tag_seq, crf.data.label_alphabet.instance2index)

    # Viterbi decoding

    # start = timeit.default_timer()
    # graph.propogate_graph()
    # end = timeit.default_timer()
    # logger.debug("propograte graph: " + str(end - start))
    print("program finished")
