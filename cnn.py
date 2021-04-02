# -*- coding: utf-8 -*-

import torch
from torchtext import data
from torchtext import datasets
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import spacy
from proj2_helpers import *


SEED = 36
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#PREPARING DATASETS

# batch_first = True because it is the form a CNN required
TEXT = data.Field(tokenize = 'spacy', batch_first = True)
LABEL = data.LabelField(dtype = torch.float)
fields = [('text', TEXT), ('label', LABEL)]

train_data, test_data = data.TabularDataset.splits(
                                        path = '/content/gdrive/My Drive/ml_project2/Datasets/',
                                        train = 'train_full_all.csv',
                                        test='test.csv',
                                        format = 'csv',
                                        fields = fields,
                                        skip_header = True
)

train_data, valid_data = train_data.split(split_ratio=0.85, random_state = random.seed(SEED))

#Vocab creation + pre-trained word embeddings
MAX_VOCAB_SIZE = 25_000
TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.twitter.27B.200d", 
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)

BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator,test_iterator = data.BucketIterator.splits(
    (train_data, valid_data,test_data), 
    batch_size = BATCH_SIZE, 
    sort_key = lambda x: x.text,
    device = device)

# MODEL CONSTRUCTION

class CNN1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.fc(cat)


class CNN2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 pad_idx):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc1 = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.fc2 = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat1 = self.dropout1(torch.cat(pooled, dim = 1))
        cat2 = self.dropout2(torch.cat(pooled, dim = 1))
        out1 = F.sigmoid(self.fc1(cat1))
        out2 = F.relu(self.fc2(cat2))
        preds = out1.mul(out2)
        return preds


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 200
HIDDEN_DIM = 100
N_FILTERS = 100 
FILTER_SIZES = [2,3,4,5]
OUTPUT_DIM = 1
N_LAYERS = 2
DROPOUT = 0.3
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN1(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
model2 = CNN2(INPUT_DIM,EMBEDDING_DIM,200,FILTER_SIZES, OUTPUT_DIM,PAD_IDX)


# Preprocess the embedding layer (add pre-trained vectors + initial weights to zero + padding tokens
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)


N_EPOCHS = 3
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'cnn-model.pt') 
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


def predict_sentiment(model, sentence, min_len = 5):
    nlp = spacy.load('en_core_web_sm')
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

#PREDICTION
with open('/content/gdrive/My Drive/ml_project2/Datasets/test_data.txt', 'r') as f:
    test_sample = f.readlines()
pred_test = []
for text in test_sample:
  pred_test.append(predict_sentiment(model,text))

ids, y_pred = reformat(pred_test)
create_csv_submission(ids, y_pred,'/content/gdrive/My Drive/ml_project2/Datasets/cnn_glove_tweet_200dim_100filters_3epochs_full.csv')
