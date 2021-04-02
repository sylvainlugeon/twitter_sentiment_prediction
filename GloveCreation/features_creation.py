#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from glove_helpers import *
import glove_solution as gs

# loading vectorized words and vocabulary
words_vect = np.load('./Datasets/embeddings_dim200.npy')
vocab = np.load('features/vocab.pkl', allow_pickle=True)


features_creation_pretrained('Datasets/glove.twitter.27B/glove.twitter.27B.200d.txt', './Datasets/train_pos_clean_200dim.txt', 'out')
features_creation_opt('Datasets/test_data.txt', './Datasets/test_features_dim200', words_vect, vocab, common_words=False, pond=1)
features_creation_opt('Datasets/train_pos.txt', './Datasets/pos_features_dim200', words_vect, vocab, common_words=False, pond=1)
features_creation_opt('Datasets/train_neg.txt', './Datasets/neg_features_dim200', words_vect, vocab)





