from helper import *
from preprocess import preprocess_label, preprocess_unlabel
from graph_tool import *
import argparse
import sklearn_crfsuite
from collections import Counter
import timeit


def read_data(file):

    with open(file) as f:
        raw_data = f.readlines()

    return raw_data


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_file", default='./data/labeled_train')
    parser.add_argument("--unlabeled_file", default='./data/unlabeled_train')
    args = parser.parse_args()

    # load dataset
    label_data = read_data(args.labeled_file)
    # unlabel_data = read_data(args.unlabeled_file)

    label_data = preprocess_label(label_data)
    test_data = label_data[0:400]
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

    # compute pmi_vectors
    pmi = PMI(ngrams_list, features_list)
    features_dict = features2dict(features_list)

    start = timeit.default_timer()
    pmi_vectors = np.array([pmi.pmi_vector(ngram, features_dict) for ngram in ngrams_list])
    end = timeit.default_timer()

    print(end-start)
    '''
    # compute CRF
    crf = sklearn_crfsuite.CRF(
        algorithm='l2sgd',
        c2=0.01,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(label_feature, label_target)
    '''