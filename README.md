### Introduction
This is the research project of my 2020 summer usra. This project mainly replicate the result of [Efficient Graph-Based Semi-Supervised Learning
of Structured Tagging Models](https://www.aclweb.org/anthology/D10-1017.pdf) and also replaced the semi-crf training part in the paper to a state-of-art [neural crf mode](https://arxiv.org/pdf/1707.06799.pdf) which pass the word emmbeddings of data into a CNN and LSTM to get input for the crf training. 

### Data
sample labeled data: PennTreebank-WSJ 

sample unlabeled data: PubMed 

### TODO:
1. finish cross-domain adaptation work
2. improve the model's performance

# ##Train:
1. training parameters are in train_config.yaml
2. To run the demo simply `python main.py` after install all requirements with `pip install -requirements.txt`

