import argparse
# import torch

from utils.data import Data
from utils.alphabet import Alphabet
from utils.functions import *


def data_initialization(data):
    data.initial_feature_alphabets()
    # data.build_alphabet(data.train_dir)
    # data.build_alphabet(data.dev_dir)
    # data.build_alphabet(data.test_dir)
    # data.fix_alphabet()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    parser.add_argument('--config', help='Configuration File', default='None')
    parser.add_argument('--wordemb', help='Embedding for words', default='cache/')
    parser.add_argument('--charemb', help='Embedding for chars', default='None')
    parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="data/model/saved_model.lstmcrf.")
    parser.add_argument('--savedset', help='Dir of saved data setting')
    parser.add_argument('--train', default="sample_data/train.bmes")
    parser.add_argument('--dev', default="sample_data/dev.bmes")
    parser.add_argument('--test', default="sample_data/test.bmes")
    parser.add_argument('--seg', default="True")
    # parser.add_argument('--raw')
    # parser.add_argument('--loadmodel')
    # parser.add_argument('--output')

    args = parser.parse_args()

    data = Data()
    data.train_dir = args.train
    data.dev_dir = args.dev
    data.test_dir = args.test
    data.model_dir = args.savemodel
    # data.dset_dir = args.savedset
    # print("Save dset directory:", data.dset_dir)

    data.word_emb_dir = args.wordemb
    data.char_emb_dir = args.charemb
    if args.seg.lower() == 'true':
        data.seg = True
    else:
        data.seg = False
    print("Seed num:", seed_num)
    # data.HP_gpu = torch.cuda.is_available()

    if args.status == 'train':
        print("MODEL: train")
        data_initialization(data)
        # data.generate_instance('train')
        # data.generate_instance('dev')
        # data.generate_instance('test')
        # data.build_pretrain_emb()
        # train(data)
