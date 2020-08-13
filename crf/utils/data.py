# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-14 17:34:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-01-25 20:25:59

from .alphabet import Alphabet
from .functions import *
from pathlib import Path
import pickle as pickle
import logging
import yaml

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"

logger = logging.getLogger("data")


class Data:
    def __init__(self):
        self.sentence_classification = False
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', True)

        self.feature_name = []
        self.feature_alphabets = []
        self.feature_num = len(self.feature_alphabets)
        self.feat_config = None

        self.tagScheme = "NoSeg"  ## BMES/BIO
        self.split_token = ' ||| '
        self.seg = True

        ### I/O
        self.train_dir = None
        self.dev_dir = None
        self.test_dir = None
        self.raw_dir = None

        self.decode_dir = None
        self.dset_dir = None  ## data vocabulary related file
        self.model_dir = None  ## model save  file
        self.load_model_dir = None  ## model load file

        self.word_emb_dir = None
        self.char_emb_dir = None
        self.feature_emb_dirs = []

        self.train_texts = []
        self.l_train_texts = []
        self.u_train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.l_train_Ids = []
        self.u_train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_feature_embeddings = []

        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        self.feature_alphabet_sizes = []
        self.feature_emb_dims = []
        self.norm_feature_embs = []
        self.word_emb_dim = 50
        self.char_emb_dim = 30

        ## Networks
        self.word_feature_extractor = "LSTM"  ## "LSTM"/"CNN"/"GRU"/
        self.use_char = True
        self.char_feature_extractor = "CNN"  ## "LSTM"/"CNN"/"GRU"/None
        self.use_crf = True
        self.nbest = None
        self.sentence_classification = False

        ## Training
        self.average_batch_loss = False
        self.optimizer = "SGD"  ## "SGD"/"AdaGrad"/"AdaDelta"/"RMSProp"/"Adam"
        self.status = "train"
        ## Hyperparameters
        self.HP_cnn_layer = 4
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True

        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_momentum = 0
        self.HP_l2 = 1e-8

    def show_data_summary(self):

        logger.info("++" * 50)
        logger.info("DATA SUMMARY START:")
        logger.info(" I/O:")
        logger.info("     Start   Sequence   Laebling   task...")
        logger.info("     Tag          scheme: %s" % (self.tagScheme))
        logger.info("     Split         token: %s" % (self.split_token))
        logger.info("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        logger.info("     MAX   WORD   LENGTH: %s" % (self.MAX_WORD_LENGTH))
        logger.info("     Number   normalized: %s" % (self.number_normalized))
        logger.info("     Word  alphabet size: %s" % (self.word_alphabet_size))
        logger.info("     Char  alphabet size: %s" % (self.char_alphabet_size))
        logger.info("     Label alphabet size: %s" % (self.label_alphabet_size))
        logger.info("     Word embedding  dir: %s" % (self.word_emb_dir))
        logger.info("     Char embedding  dir: %s" % (self.char_emb_dir))
        logger.info("     Word embedding size: %s" % (self.word_emb_dim))
        logger.info("     Char embedding size: %s" % (self.char_emb_dim))
        logger.info("     Norm   word     emb: %s" % (self.norm_word_emb))
        logger.info("     Norm   char     emb: %s" % (self.norm_char_emb))
        logger.info("     Train  file directory: %s" % (self.train_dir))
        logger.info("     Dev    file directory: %s" % (self.dev_dir))
        logger.info("     Test   file directory: %s" % (self.test_dir))
        logger.info("     Raw    file directory: %s" % (self.raw_dir))
        logger.info("     Dset   file directory: %s" % (self.dset_dir))
        logger.info("     Model  file directory: %s" % (self.model_dir))
        logger.info("     Loadmodel   directory: %s" % (self.load_model_dir))
        logger.info("     Decode file directory: %s" % (self.decode_dir))
        logger.info("     Train instance number: %s" % (len(self.train_texts)))
        logger.info("     Dev   instance number: %s" % (len(self.dev_texts)))
        logger.info("     Test  instance number: %s" % (len(self.test_texts)))
        logger.info("     Raw   instance number: %s" % (len(self.raw_texts)))
        logger.info("     FEATURE num: %s" % (self.feature_num))
        for idx in range(self.feature_num):
            logger.info("         Fe: %s  alphabet  size: %s" % (
                self.feature_alphabets[idx].name, self.feature_alphabet_sizes[idx]))
            logger.info(
                "         Fe: %s  embedding  dir: %s" % (self.feature_alphabets[idx].name, self.feature_emb_dirs[idx]))
            logger.info(
                "         Fe: %s  embedding size: %s" % (self.feature_alphabets[idx].name, self.feature_emb_dims[idx]))
            logger.info(
                "         Fe: %s  norm       emb: %s" % (self.feature_alphabets[idx].name, self.norm_feature_embs[idx]))
        logger.info(" " + "++" * 20)
        logger.info(" Model Network:")
        logger.info("     Model        use_crf: %s" % (self.use_crf))
        logger.info("     Model word extractor: %s" % (self.word_feature_extractor))
        logger.info("     Model       use_char: %s" % (self.use_char))
        if self.use_char:
            logger.info("     Model char extractor: %s" % (self.char_feature_extractor))
            logger.info("     Model char_hidden_dim: %s" % (self.HP_char_hidden_dim))
        logger.info(" " + "++" * 20)
        logger.info(" Training:")
        logger.info("     Optimizer: %s" % (self.optimizer))
        logger.info("     Iteration: %s" % (self.HP_iteration))
        logger.info("     BatchSize: %s" % (self.HP_batch_size))
        logger.info("     Average  batch   loss: %s" % (self.average_batch_loss))

        logger.info(" " + "++" * 20)
        logger.info(" Hyperparameters:")

        logger.info("     Hyper              lr: %s" % (self.HP_lr))
        logger.info("     Hyper        lr_decay: %s" % (self.HP_lr_decay))
        logger.info("     Hyper         HP_clip: %s" % (self.HP_clip))
        logger.info("     Hyper        momentum: %s" % (self.HP_momentum))
        logger.info("     Hyper              l2: %s" % (self.HP_l2))
        logger.info("     Hyper      hidden_dim: %s" % (self.HP_hidden_dim))
        logger.info("     Hyper         dropout: %s" % (self.HP_dropout))
        logger.info("     Hyper      lstm_layer: %s" % (self.HP_lstm_layer))
        logger.info("     Hyper          bilstm: %s" % (self.HP_bilstm))
        logger.info("     Hyper             GPU: %s" % (self.HP_gpu))
        logger.info("DATA SUMMARY END.")
        logger.info("++" * 50)
        sys.stdout.flush()

    def initial_feature_alphabets(self):
        # not used in this project
        items = open(self.train_dir, 'r').readline().strip('\n').split()
        total_column = len(items)

        # Add more features
        if total_column > 2:
            for idx in range(1, total_column - 1):
                feature_prefix = items[idx].split(']', 1)[0] + "]"
                self.feature_alphabets.append(Alphabet(feature_prefix))
                self.feature_name.append(feature_prefix)
                logger.info("Find feature: ", feature_prefix)

        self.feature_num = len(self.feature_alphabets)
        self.pretrain_feature_embeddings = [None] * self.feature_num
        self.feature_emb_dims = [20] * self.feature_num
        self.feature_emb_dirs = [None] * self.feature_num
        self.norm_feature_embs = [False] * self.feature_num
        self.feature_alphabet_sizes = [0] * self.feature_num
        if self.feat_config:
            for idx in range(self.feature_num):
                if self.feature_name[idx] in self.feat_config:
                    self.feature_emb_dims[idx] = self.feat_config[self.feature_name[idx]]['emb_size']
                    self.feature_emb_dirs[idx] = self.feat_config[self.feature_name[idx]]['emb_dir']
                    self.norm_feature_embs[idx] = self.feat_config[self.feature_name[idx]]['emb_norm']

    def build_alphabet(self, input_data_list, content):
        if content == "text":
            for input_data in input_data_list:
                for sent in input_data:
                    for word in sent:
                        word = normalize_word(word)
                        self.word_alphabet.add(word)
                        ## build feature alphabet
                        # for idx in range(self.feature_num):
                        #     feat_idx = pairs[idx + 1].split(']', 1)[-1]
                        #     self.feature_alphabets[idx].add(feat_idx)
                        for char in word:
                            self.char_alphabet.add(char)

            self.word_alphabet_size = self.word_alphabet.size()
            self.char_alphabet_size = self.char_alphabet.size()

        elif content == "label":
            for input_data in input_data_list:
                for sent in input_data:
                    for label in sent:
                        self.label_alphabet.add(label)

            self.label_alphabet_size = self.label_alphabet.size()

        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        if self.sentence_classification:
            self.tagScheme = "Not sequence labeling task"

        for idx in range(self.feature_num):
            self.feature_alphabet_sizes[idx] = self.feature_alphabets[idx].size()

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        for idx in range(self.feature_num):
            self.feature_alphabets[idx].close()

    def build_pretrain_emb(self):
        if self.word_emb_dir:
            logger.info("Load pretrained word embedding, norm: %s, dir: %s" % (self.norm_word_emb, self.word_emb_dir))
            self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(self.word_emb_dir,
                                                                                       self.word_alphabet,
                                                                                       self.word_emb_dim,
                                                                                       self.norm_word_emb)
        if self.char_emb_dir:
            logger.info("Load pretrained char embedding, norm: %s, dir: %s" % (self.norm_char_emb, self.char_emb_dir))
            self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(self.char_emb_dir,
                                                                                       self.char_alphabet,
                                                                                       self.char_emb_dim,
                                                                                       self.norm_char_emb)
        for idx in range(self.feature_num):
            if self.feature_emb_dirs[idx]:
                logger.info("Load pretrained feature %s embedding:, norm: %s, dir: %s" % (
                    self.feature_name[idx], self.norm_feature_embs[idx], self.feature_emb_dirs[idx]))
                self.pretrain_feature_embeddings[idx], self.feature_emb_dims[idx] = build_pretrain_embedding(
                    self.feature_emb_dirs[idx], self.feature_alphabets[idx], self.feature_emb_dims[idx],
                    self.norm_feature_embs[idx])

    def get_instance(self, name, texts, labels=None):
        if name == "labeled_train":
            self.l_train_texts, self.l_train_Ids = gene_instance(texts, labels, self.word_alphabet,
                                                                     self.char_alphabet,
                                                                     self.feature_alphabets, self.label_alphabet,
                                                                     self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                                     self.split_token)
        elif name == "unlabeled_train":
            self.u_train_texts, self.u_train_Ids = gene_instance(texts, labels, self.word_alphabet, self.char_alphabet,
                                                      self.feature_alphabets, self.label_alphabet,
                                                      self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                      self.split_token)
        elif name == "train":
            self.train_texts, self.train_Ids = read_instance(self.train_dir, self.word_alphabet, self.char_alphabet,
                                                             self.feature_alphabets, self.label_alphabet,
                                                             self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                             self.split_token)
        else:
            logger.info("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))

    def generate_instance(self, name):
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(self.train_dir, self.word_alphabet, self.char_alphabet,
                                                             self.feature_alphabets, self.label_alphabet,
                                                             self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                             self.sentence_classification, self.split_token)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance(self.dev_dir, self.word_alphabet, self.char_alphabet,
                                                         self.feature_alphabets, self.label_alphabet,
                                                         self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                         self.sentence_classification, self.split_token)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(self.test_dir, self.word_alphabet, self.char_alphabet,
                                                           self.feature_alphabets, self.label_alphabet,
                                                           self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                           self.sentence_classification, self.split_token)
        else:
            logger.info("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))

    def write_decoded_results(self, predict_results, name):

        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            logger.info("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        fout = open(self.decode_dir, 'w')
        for idx in range(sent_num):
            if self.sentence_classification:
                fout.write(" ".join(content_list[idx][0]) + "\t" + predict_results[idx] + '\n')
            else:
                sent_length = len(predict_results[idx])
                for idy in range(sent_length):
                    ## content_list[idx] is a list with [word, char, label]
                    fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')
                fout.write('\n')
        fout.close()
        logger.info("Predict %s result has been written into file. %s" % (name, self.decode_dir))

    def load(self, data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def write_nbest_decoded_results(self, predict_results, pred_scores, name):
        ## predict_results : [whole_sent_num, nbest, each_sent_length]
        ## pred_scores: [whole_sent_num, nbest]
        fout = open(self.decode_dir, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            logger.info("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        assert (sent_num == len(pred_scores))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx][0])
            nbest = len(predict_results[idx])
            score_string = "# "
            for idz in range(nbest):
                score_string += format(pred_scores[idx][idz], '.4f') + " "
            fout.write(score_string.strip() + "\n")

            for idy in range(sent_length):
                try:  # Will fail with python3
                    label_string = content_list[idx][0][idy].encode('utf-8') + " "
                except:
                    label_string = content_list[idx][0][idy] + " "
                for idz in range(nbest):
                    label_string += predict_results[idx][idz][idy] + " "
                label_string = label_string.strip() + "\n"
                fout.write(label_string)
            fout.write('\n')
        fout.close()
        logger.info("Predict %s %s-best result has been written into file. %s" % (name, nbest, self.decode_dir))

    def read_config(self, config_file):
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.Loader)
        ## read data:

        for key in config.keys():
            if hasattr(self, key):
                if type(config[key]) == str and '/' in config[key]:
                    setattr(self, key, Path(config[key]))
                else:
                    setattr(self, key, config[key])

    def str2bool(string):
        if string == "True" or string == "true" or string == "TRUE":
            return True
        else:
            return False
