import os
import pickle
import torch


def save_ins(ins, ins_name, save_dir):
    instance_file_name = ins_name + '.pkl'
    instance_dir = save_dir + instance_file_name
    f = open(instance_dir, "wb")
    pickle.dump(ins, f)
    f.close()


def load_ins(ins_name, save_dir):
    file_name = ins_name + '.pkl'
    instance_dir = save_dir + file_name
    f = open(instance_dir, "rb")
    loaded_instance = pickle.load(f)
    return loaded_instance


def save_tensor(ins, ins_name, save_dir):
    instance_file_name = ins_name + '.pt'
    instance_dir = save_dir + instance_file_name
    torch.save(ins, instance_dir)


def load_tensor(ins_name, save_dir):
    file_name = ins_name + '.pt'
    instance_dir = save_dir + file_name
    return torch.load(instance_dir)

