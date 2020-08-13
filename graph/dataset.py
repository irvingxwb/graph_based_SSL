from .functions import sent2graphfeatures


def preprocess_data(raw_data, mode):
    text = []
    label = []
    text_sent = []
    label_sent = []
    if mode == "labeled":
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

    elif mode == "unlabeled":
        for line in raw_data:
            if line == '\n' and len(text_sent) != 0:
                text.append(text_sent)
                text_sent = []
            else:
                sent = line.split()
                text_sent.append(sent[0])
        if len(text_sent) != 0:
            text.append(text_sent)

    return text, label


class Dataset:
    # I/O
    word_emb_dir = None
    labeled_train_dir = None
    unlabeled_train_dir = None

    # data
    train_texts = None
    labeled_train_texts = None
    labeled_train_labels = None
    unlabeled_train_texts = None

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

        # select and combine dataset
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
