# -*- coding: utf-8 -*-
import torch
import random
import spacy
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from proj2_helpers import *

SEED = 36
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#PREPARING DATASETS

# Load training and testing dataset
TEXT = data.Field(tokenize = 'spacy', preprocessing = generate_bigrams)
LABEL = data.LabelField(dtype = torch.float,sequential=False, use_vocab=False)
fields = [('text', TEXT), ('label', LABEL)]

train_data, test_data = data.TabularDataset.splits(
                                        path = '/content/gdrive/My Drive/ml_project2/Datasets/',
                                        train = 'train_full_all.csv',
                                        test='test.csv',
                                        format = 'csv',
                                        fields = fields,
                                        skip_header = True
)


train_data, valid_data = train_data.split(random_state = random.seed(SEED))


#Vocab creation + pre-trained word embeddings
MAX_VOCAB_SIZE = 25_000
TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.twitter.27B.200d")

LABEL.build_vocab(train_data)


BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator,test_iterator = data.BucketIterator.splits(
    (train_data, valid_data,test_data), 
    batch_size = BATCH_SIZE, 
    sort_key = lambda x: x.text,
    device = device)

# MODEL CONSTRUCTION

class FastText1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, text):
        
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        return self.fc(pooled)


class FastText2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        hidden = self.fc1(pooled)
        act2 = F.avg_pool2d(hidden)
        output = self.fc2(act2)  
        return output


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 200
HIDDEN_DIM=2
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
model = FastText1(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
model2 = FastText2(INPUT_DIM,EMBEDDING_DIM,HIDDEN_DIM,OUTPUT_DIM,PAD_IDX)

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

N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'fastText-model.pt') 
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


#PREDICTION

def predict_sentiment(model, sentence):
    model.eval()
    nlp = spacy.load('en_core_web_sm')
    tokenized = generate_bigrams([tok.text for tok in nlp.tokenizer(sentence)])
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

with open('/content/gdrive/My Drive/ml_project2/Datasets/test_data.txt', 'r') as f:
    test_sample = f.readlines()
pred_test = []
for text in test_sample:
  pred_test.append(predict_sentiment(model,text))

ids,y_pred = reformat(pred_test)

create_csv_submission(ids, y_pred,'/content/gdrive/My Drive/ml_project2/Datasets/fast_twitter_glove200dim_batch128_full_5epochs.csv')
