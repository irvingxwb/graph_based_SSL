from .functions import sent2graphfeatures


def preprocess_data(raw_data, mode):
    text = []
    label = []
    text_sent = []
    label_sent = []

    if mode == "unlabeled":
        for line in raw_data:
            if line == '\n' and len(text_sent) != 0:
                text.append(text_sent)
                text_sent = []
            else:
                sent = line.split()
                text_sent.append(sent[0])
        if len(text_sent) != 0:
            text.append(text_sent)
    else:
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


class Dataset:
    # I/O
    word_emb_dir = None
    labeled_train_dir = None
    unlabeled_train_dir = None
    dev_dir = None
    test_dir = None

    # data
    train_texts = None
    labeled_train_texts = None
    labeled_train_labels = None
    unlabeled_train_texts = None
    dev_texts = None
    dev_labels = None
    test_texts = None
    test_labels = None

    # hyper parameters
    k_nearest = 5
    unlabeled_num = 0
    gpu = False

    # number
    labeled_cnt = 0
    unlabeled_cnt = 0

    def __init__(self):
        pass

    def load_all_data(self):
        if self.labeled_train_dir:
            with open(self.labeled_train_dir) as f:
                raw_data = f.readlines()
            self.labeled_train_texts, self.labeled_train_labels = preprocess_data(raw_data, 'labeled')

        if self.unlabeled_train_dir:
            with open(self.unlabeled_train_dir) as f:
                raw_data = f.readlines()
            self.unlabeled_train_texts, _ = preprocess_data(raw_data, "unlabeled")

        if self.dev_dir:
            with open(self.dev_dir) as f:
                raw_data = f.readlines()
            self.dev_texts, self.dev_labels = preprocess_data(raw_data, 'dev')

        if self.test_dir:
            with open(self.test_dir) as f:
                raw_data = f.readlines()
            self.test_texts, self.test_labels = preprocess_data(raw_data, 'test')

        # select and combine dataset
        l_select_num = 1000
        u_select_num = 1000
        self.labeled_train_texts = self.labeled_train_texts[0:l_select_num]
        self.labeled_train_labels = self.labeled_train_labels[0:l_select_num]
        self.unlabeled_train_texts = self.unlabeled_train_texts[0:u_select_num]
        self.train_texts = self.labeled_train_texts + self.unlabeled_train_texts
        self.labeled_cnt = len(self.labeled_train_texts)
        self.unlabeled_cnt = len(self.unlabeled_train_texts)

    def get_features_list(self):
        features_list = []
        for sent in self.labeled_train_texts:
            features = sent2graphfeatures(sent)
            features_list.extend(features)

        return features_list

    # todo: produce word emb instead of using pre-trained
    def build_word_emb(self):
        pass
