from pathlib import Path
import timeit
import string
import re


def read_labeled(pathlist):
    data = []
    sent = []
    for file in pathlist:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '')
                if line == '' or line[0] == '=':
                    if len(sent) >= 10:
                        data.append(sent)
                        sent = []
                else:
                    line = line.replace('[', '')
                    line = line.replace(']', '')
                    words = line.split(' ')
                    while '' in words:
                        words.remove('')

                    for word in words:
                        if word is not '':
                            sent.append(word.split('/'))

    return data


def write_labeled_data(name, data):
    l_dest_dir = "../data/train/labeled_" + name
    with open(l_dest_dir, mode="w+") as f:
        for sent in data:
            for word in sent:
                f.write("%s %s\n" % (word[0], word[1]))
            f.write("\n")


def read_unlabeled(pathlist):
    data = []
    sent = []
    for file in pathlist:
        with open(file, 'r', encoding='utf8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '')
                line = re.sub(r'\(.*\)', '', line)
                line = line.replace(',', ' , ')
                line = line.replace('.', ' . ')

                if line == '':
                    continue
                else:
                    words = line.split(' ')
                    while '' in words:
                        words.remove('')

                    for word in words:
                        sent.append(word)
                        if word == '.':
                            if len(sent) > 10:
                                data.append(sent)
                            sent = []

                    if len(sent) > 10:
                        data.append(sent)
                    sent = []
    return data


def write_unlabeled_data(data):
    u_dest_dir = "../data/train/unlabeled_train"
    file_name = u_dest_dir
    with open(file_name, mode="w+") as f:
        for sent in data:
            for word in sent:
                f.write("%s\n" % word)
            f.write("\n")


def process_labeled(name, data_start, data_end):
    l_data_dir = "../data/source/PennTreeBank"
    pathlist = Path(l_data_dir).rglob('*.pos')

    start = timeit.default_timer()
    data = read_labeled(pathlist)
    end = timeit.default_timer()
    print("reading labeled time cost : %d \ndata length %d" % ((end - start), len(data)))
    sub_data = data[data_start:data_end]
    write_labeled_data(name, sub_data)

    print("finish process labeled data for : %s" % name)


def process_unlabeled(data_length):
    u_data_dir = "../data/source/PubMed"
    pathlist = Path(u_data_dir).rglob('*.txt')
    file_number = 1
    # count unlabeled file
    start = timeit.default_timer()
    data = read_unlabeled(pathlist)
    end = timeit.default_timer()
    print("reading unlabeled time cost : %d \ndata length %d" % ((end - start), len(data)))

    if end > len(data):
        end = len(data)
    sub_data = data[0:data_length]
    write_unlabeled_data(sub_data)

    print("finish process unlabled data")


if __name__ == "__main__":
    # labeled_data length around 50000
    t_start = 0
    t_end = 4000
    v_start = 5000
    v_end = 7000
    te_start = 8000
    te_end = 10000
    unlabeled_length = 50000
    process_labeled('train', t_start, t_end)
    process_labeled('dev', v_start, v_end)
    process_labeled('test', te_start, te_end)
    process_unlabeled(unlabeled_length)
