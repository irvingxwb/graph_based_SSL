from helper import *
from preprocess import preprocess_label, preprocess_unlabel
from graph_tool import *
import argparse
import sklearn_crfsuite
import timeit
from sys import getsizeof
import logging
logging.basicConfig(level=logging.DEBUG, filename='./main.log', filemode='w', format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

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
    label_data = read_data(args.labeled_file)
    # unlabel_data = read_data(args.unlabeled_file)
    unlabeledNum = 0

    label_data = preprocess_label(label_data)
    test_data = label_data
    logger.debug("length of label_data: "+ str(len(label_data)))
    # unlabel_data = preprocess_unlabel(unlabel_data)
    # all_data = label_data + unlabel_data

    # extract features
    label_feature = [sent2features(s) for s in test_data]
    label_target = [sent2labels(s) for s in test_data]

    # create graph with labeled and unlabeled data
    ngrams_list = []
    for sent in test_data:
        ngrams = sent2trigrams(sent)
        ngrams_list.extend(ngrams)

    features_list = []
    for sent in test_data:
        features = sent2graphfeatures(sent)
        features_list.extend(features)

    start = timeit.default_timer()
    pmi = PMI(ngrams_list, features_list)
    end = timeit.default_timer()
    logger.debug("Construct PMI: "+ str(end - start))

    # start = timeit.default_timer()
    # pmi_vectors_sparse = pmi.pmi_vectors_sparse()
    # end = timeit.default_timer()
    # logging.debug("vectors sparse: "+ str(end - start)+" "+str(getsizeof(pmi_vectors_sparse)))

    start = timeit.default_timer()
    pmi_vectors_improve = pmi.pmi_vectors_improve()
    end = timeit.default_timer()
    logger.debug("vectors improve: "+ str(end - start))

    # construct graph
    start = timeit.default_timer()
    graph = Graph(list(pmi.ngrams_feature_map.keys()), pmi_vectors_improve, unlabeledNum, k_nearest)
    end = timeit.default_timer()
    logger.debug("Construct Graph: "+ str(end - start))

    # compute CRF
    crf = sklearn_crfsuite.CRF(
        algorithm='l2sgd',
        c2=0.01,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(label_feature, label_target)
    marginal_prob = crf.predict_marginals(label_feature)

    marginal_prob_agg = agg_marginal(marginal_prob, ngrams_list, pmi.ngrams_counter)

    print("program finished")


