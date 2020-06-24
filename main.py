from helper import *
from preprocess import preprocess_label, preprocess_unlabel
from feature import FeatureSet
from crf import LinearChainCRF
from graph import *
from pmi import *
import argparse
import sklearn_crfsuite
import timeit
from sys import getsizeof, stdout
import logging

# logging.basicConfig(level=logging.DEBUG, filename='./main.log', filemode='w', format='%(asctime)s:%(levelname)s:%(
# name)s:%(message)s')
logging.basicConfig(level=logging.DEBUG, stream=stdout, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logger = logging.getLogger('Main')
# define hyperparameter
k_nearest = 3

def read_data(file):
    with open(file) as f:
        raw_data = f.readlines()

    return raw_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_file", default='./data/labeled_train')
    parser.add_argument("--unlabeled_file", default='./data/unlabeled_train')
    args = parser.parse_args()
    # load dataset
    logger.debug("start reading data")
    label_data = read_data(args.labeled_file)
    # unlabel_data = read_data(args.unlabeled_file)
    unlabeledNum = 0

    label_data = preprocess_label(label_data)
    label_data = label_data[0:8000]
    logger.debug("length of label_data: "+ str(len(label_data)))
    # unlabel_data = preprocess_unlabel(unlabel_data)
    # all_data = label_data + unlabel_data

    # extract features
    label_feature = [sent2features(sent) for sent in label_data]
    label_target = [sent2targets(sent) for sent in label_data]

    # create graph with labeled and unlabeled data
    ngrams_list = []
    for sent in label_data:
        ngrams = sent2trigrams(sent)
        ngrams_list.extend(ngrams)

    features_list = []
    for sent in label_data:
        features = sent2graphfeatures(sent)
        features_list.extend(features)

    logger.debug("preprocess data")
    featureSet = FeatureSet(sent_set=label_data, labels_set=label_target)
    logger.debug("Construct FeatureSet")

    crf = LinearChainCRF(feature_set=featureSet, training_data=label_data)
    crf.train()

    # pmi = PMI(ngrams_list, features_list)
    # logger.debug("Construct PMI")
    #
    # pmi_vectors_improve = pmi.pmi_vectors_improve()
    # logger.debug("vectors improve")
    #
    # # construct graph
    # start = timeit.default_timer()
    # graph = Graph(list(pmi.ngrams_feature_map.keys()), pmi_vectors_improve, unlabeledNum, k_nearest)
    # end = timeit.default_timer()
    # logger.debug("Construct Graph: "+ str(end - start))
    #
    # # posterior decoding
    # crf = sklearn_crfsuite.CRF(
    #     algorithm='l2sgd',
    #     c2=0.01,
    #     max_iterations=100,
    #     all_possible_transitions=True
    # )
    # crf.fit(label_feature, label_target)
    # marginal_prob = crf.predict_marginals(label_feature)
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


