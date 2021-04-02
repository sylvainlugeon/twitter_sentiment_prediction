# -*- coding: utf-8 -*-
"""some helper functions for project 2."""
import csv
import numpy as np
import torch
import time




def create_csv_submission(ids, y_pred,name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w+', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
    

# Function for the model

def generate_bigrams(x):
    """
    genere the bi-gram of the set each each tweet : split it by sequences of 
    two words/characters (ex: "I love Machine Learning": ["I love", "love Machine",
    "Machine Learning"])
    """
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc    

def train(model, iterator, optimizer, criterion):
    """
    train our model using back propagation
    """
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    """
    evaluate our model without forgetting to disable the back propagation
    we should not update anymore our weights
    """
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    """
    auxiliary function to see the time
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def reformat(pred_test):
    """
    reformat the model output to fit our submission format 
    (list of -1 or 1)
    """
    idx_test = list(range(1,len(pred_test)+1))
    predicted = []
    for pred in pred_test:
        if pred <0.5:
            predicted.append(-1)
        else:
            predicted.append(1)
    return idx_test, predicted
