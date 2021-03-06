import argparse
import torch
import gc
import torch.nn as nn
import torch.optim as optim
from sys import stdout
import logging
import time
import random

from .utils.data import Data
from .utils.alphabet import Alphabet
from .utils.metric import get_ner_fmeasure
from .utils.functions import *
from .model.seqlabel import SeqLabel
from graph.dataset import Dataset

# logging.basicConfig(level=logging.DEBUG, filename='./log/main.log', filemode='w', format='%(asctime)s:%(levelname)s:%(
# name)s:%(message)s')
logging.basicConfig(level=logging.DEBUG, stream=stdout, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('main')


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    logger.info(" Learning rate is set as: %s" % str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)

    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()

    # logger.info("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)

    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []

    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)

    return pred_label, gold_label


def recover_label_probs(tag_seq, tag_probs, mask_variable, word_recover):
    tag_seq = tag_seq[word_recover]
    tag_probs = tag_probs[word_recover]
    mask_variable = mask_variable[word_recover]

    return tag_seq, tag_probs, mask_variable


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if
                         mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def evaluate(data, model, name, nbest=None):
    if name == "train":
        instances = data.l_train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        logger.info("Error: wrong evaluate name,", name)
        exit(1)
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
            instance, data.HP_gpu, False)
        if nbest and not data.sentence_classification:
            scores, nbest_tag_seq = model.decode_nbest(batch_word, batch_features, batch_wordlen, batch_char,
                                                       batch_charlen, batch_charrecover, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:, :, 0]
        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
                            mask)
        # logger.info("tag:",tag_seq)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time

    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)

    return speed, acc, p, r, f, pred_results, pred_scores


def batchify_with_label(input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # logger.info len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask.type(
        torch.bool)


def train(data):
    logger.info("Training model...")
    # save_data_name = data.model_dir + ".dset"
    # data.save(save_data_name)

    model = SeqLabel(data)

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        logger.info("Optimizer illegal: %s" % (data.optimizer))
        exit(1)
    best_dev = -10
    # data.HP_iteration = 1
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        logger.info("Epoch: %s/%s" % (idx, data.HP_iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)

        instance_count = 0
        sample_id = 0
        sample_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.l_train_Ids)
        logger.info("Shuffle: first input word list:" + str(data.l_train_Ids[0][0]))

        ## set model in train model
        model.train()
        model.zero_grad()

        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.l_train_Ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.l_train_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask \
                = batchify_with_label(instance, data.HP_gpu, if_train=True)
            instance_count += 1
            loss, tag_seq = model.calculate_loss(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                                 batch_charrecover, batch_label, mask)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            # logger.info("loss:",loss.item())
            sample_loss += loss.item()
            total_loss += loss.item()
            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                logger.info("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                    end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    logger.info("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                sys.stdout.flush()
                sample_loss = 0
            loss.backward()
            optimizer.step()
            model.zero_grad()
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
        end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
        idx, epoch_cost, train_num / epoch_cost, total_loss))
        print("totalloss:", total_loss)
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)
        # continue
        speed, acc, p, r, f, _, _ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        current_score = acc
        print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))

        if current_score > best_dev:
            if data.seg:
                print("Exceed previous best f score:", best_dev)
            else:
                print("Exceed previous best acc score:", best_dev)
            model_name = data.model_dir + '/' + str(idx) + ".model"
            print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score
        # ## decode test
        speed, acc, p, r, f, _, _ = evaluate(data, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if data.seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc))
        gc.collect()


class NCRFpp:
    # data structure for saving crf data
    data = Data()
    model = None
    decode_name = 'raw'

    def __init__(self, config="train_config.yaml"):
        self.data.read_config(config)
        self.data.HP_gpu = torch.cuda.is_available()
        self.data.HP_l2 = float(self.data.HP_l2)
        logger.info("GPU available: " + str(self.data.HP_gpu))

    # labeled train text, unlabeled train text, labeled_train_label, test_text
    def build_crf(self, data_set):
        logger.info("MODEL: train")
        l_texts = data_set.labeled_train_texts
        u_texts = data_set.unlabeled_train_texts
        l_labels = data_set.labeled_train_labels
        d_texts = data_set.dev_texts
        d_labels = data_set.dev_labels
        t_texts = data_set.test_texts
        t_labels = data_set.test_labels
        self.data.build_alphabet([l_texts, u_texts, d_texts, t_texts], content='text')
        self.data.build_alphabet([l_labels, d_labels, t_labels], content='label')
        self.data.fix_alphabet()
        self.data.get_instance('labeled_train', l_texts, l_labels)
        self.data.get_instance('unlabeled_train', u_texts)
        self.data.get_instance('dev', d_texts, d_labels)
        self.data.get_instance('test', t_texts, t_labels)
        self.data.build_pretrain_emb()

    def train_crf(self, mode, text=None):
        if mode == "train":
            train(self.data)

    def decode_marginals(self):
        logger.info("model: decode")

        self.model = SeqLabel(self.data)
        self.model.load_state_dict(torch.load(self.data.load_model_dir))
        self.model.eval()
        batch_size = self.data.HP_batch_size

        instances = self.data.train_Ids

        # decode labeled data
        tag_seq = []
        tag_seq_probs = []
        tag_seq_mask = []

        train_num = len(instances)
        total_batch = train_num // batch_size + 1

        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = instances[start:end]
            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, \
            batch_label, mask = batchify_with_label(instance, self.data.HP_gpu, False)

            temp_seq, temp_probs = self.model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                  batch_charrecover, mask, prob=True)
            # build return values
            batch_seq, batch_probs, batch_mask = recover_label_probs(temp_seq, temp_probs, mask, batch_wordrecover)

            tag_seq += batch_seq
            tag_seq_probs += batch_probs
            tag_seq_mask += batch_mask

        return tag_seq, tag_seq_probs, tag_seq_mask

    def evaluate_tags(self, tag_seq, labels):
        match_cnt = 0
        total_cnt = 0
        for s_idx, tag_sent in enumerate(tag_seq):
            for idx, tag in enumerate(tag_sent):
                if idx < len(labels[s_idx]):
                    tag_label = self.data.label_alphabet.get_instance(tag)
                    label = labels[s_idx][idx]
                total_cnt += 1
                if label == tag_label:
                    match_cnt += 1

        acc = match_cnt / total_cnt
        return acc

    def decode_sequence(self):
        logger.info("model: decode")

        if not self.model:
            self.model = SeqLabel(self.data)

        logger.info("Load Model from file: ", str(self.data.model_dir))
        self.model.load_state_dict(torch.load(self.data.load_model_dir))

        logger.info("Decode %s data, nbest: %s ..." % (self.decode_name, self.data.nbest))
        start_time = time.time()
        speed, acc, p, r, f, pred_results, pred_scores = evaluate(self.data, self.model, self.decode_name, self.data.nbest)
        end_time = time.time()
        time_cost = end_time - start_time
        if self.data.seg:
            logger.info(
                "%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (self.decode_name, time_cost, speed,
                                                                                          acc, p, r, f))
        else:
            logger.info("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (self.decode_name, time_cost, speed, acc))

        return pred_results, pred_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    parser.add_argument('--config', help='Configuration File', default='train_config.yaml')

    args = parser.parse_args()

    data = Data()

    # read config from file
    data.read_config(args.config)
    data.HP_gpu = torch.cuda.is_available()
    data.HP_l2 = float(data.HP_l2)
    logger.info("GPU available: " + str(data.HP_gpu))

    # deprecated main function
    #
    # if data.status == 'train':
    #     logger.info("MODEL: train")
    #     data_initialization(data)
    #     data.generate_instance('train')
    #     data.generate_instance('dev')
    #     data.generate_instance('test')
    #     data.build_pretrain_emb()
    #     train(data)
