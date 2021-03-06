from graph.functions import *
from graph.graph import *
from graph.dataset import Dataset
import argparse
import torch
from crf.ncrfpp import NCRFpp
from sys import stdout
import logging

# logging.basicConfig(level=logging.DEBUG, filename='./log/main.log', filemode='w', format='%(asctime)s:%(levelname)s:%(
# name)s:%(message)s')
logging.basicConfig(level=logging.DEBUG, stream=stdout, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logger = logging.getLogger('Main')


def logging_star():
    logger.info('**' * 40)
    logger.info('**' * 40)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_train", default='./data/train/labeled_train')
    parser.add_argument("--unlabeled_train", default='./data/train/unlabeled_train')
    parser.add_argument("--dev", default='./data/train/labeled_dev')
    parser.add_argument("--test", default='./data/train/labeled_test')
    parser.add_argument("--graph_dir", default='./data/save/graph/')
    parser.add_argument("--crf_dir", default='./data/save/crf/')
    args = parser.parse_args()

    # debug mode flags, use all prestored data for debugging
    load_pmi = False
    load_graph = False
    load_crf = True

    # add crf data structure to main data set
    data_set = Dataset()
    data_set.gpu = torch.cuda.is_available()
    data_set.labeled_train_dir = args.labeled_train
    data_set.unlabeled_train_dir = args.unlabeled_train
    data_set.dev_dir = args.dev
    data_set.test_dir = args.test

    # load data set
    data_set.load_all_data()
    logger.debug("length of labeled data: %d sentences" % data_set.labeled_cnt)
    logger.debug("length of unlabeled data: %d sentences" % data_set.unlabeled_cnt)

    # initialize graph
    graph = Graph(data_set)
    if not load_pmi:
        graph.build_pmi_vectors()
        graph.save(args.graph_dir, 'pmi')
    else:
        graph.load(args.graph_dir, 'pmi')
        logger.debug("Load pre-computed pmi vectors from file")

    if not load_graph:
        graph.construct_graph()
        graph.save(args.graph_dir, 'graph')
    else:
        graph.load(args.graph_dir, 'graph')
        logger.debug("Load pre-computed graph from file")

    # initialize crf
    crf = NCRFpp()
    crf.build_crf(data_set)

    if not load_crf:
        # train crf, time consuming
        crf.train_crf("train")
        # labeled + unlabeled data
        tag_labels, tag_probs, tag_mask = crf.decode_marginals()
        # tag sequence contain unlabeled data tags, only evaluate tags with labeled data
        acc = crf.evaluate_tags(tag_labels, data_set.labeled_train_labels)
        logger.debug("decode accuracy on labeled data: %s" % str(acc))
        save_crf_result(tag_labels, tag_probs, tag_mask, args.crf_dir)
        graph.update_train_result(tag_labels, tag_probs, tag_mask)
        logger.debug("finish crf train")
    else:
        tag_labels, tag_probs, tag_mask = load_crf_result(args.crf_dir)
        acc = crf.evaluate_tags(tag_labels, data_set.labeled_train_labels)
        graph.update_train_result(tag_labels, tag_probs, tag_mask)
        logger.debug("load crf result from file")

    logger.debug("sequence length %d" % len(tag_labels))

    # token to type map
    # parameter : train text for decode
    graph.token2type_map(data_set.labeled_train_texts + data_set.unlabeled_train_texts)

    # graph propogations
    graph.graph_props(iter_num=5, label_count=crf.data.label_alphabet_size)

    # Viterbi decoding
    graph.viterbi_decode()

    # Retrain crf
    retrain_data_texts = graph.generate_retrain_data()
    # crf.train_crf(mode='retrain', text=retrain_data_texts)