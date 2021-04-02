#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from proj2_helpers import create_csv_submission


#Creation glove_features with 200 dimensions
train_neg_sample = pd.read_csv('neg_features_200', index_col=0)
train_pos_sample = pd.read_csv('pos_features_200', index_col=0)
data_test = pd.read_csv('test_features_200', index_col=0)


#Processing by droping null row
train_neg_sample.dropna(0)
train_pos_sample.dropna(0)
data_test.dropna(0)


train_all = np.concatenate((train_pos_sample, train_neg_sample), axis=0)
train_all_target = np.concatenate((np.ones(len(train_pos_sample)), np.zeros(len(train_neg_sample))),axis=0) 

#-------------------------------------------------------------------------------------------------------------------

print("svm processing")
#Text classifier with pipeline for SVM
text_clf = Pipeline([('tfidf', TfidfTransformer(norm='l2')),('clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter= 10, tol=1e-3,epsilon=0.1, eta0=0.0,learning_rate='optimal', random_state=42))])

#Grid search for parameters of SVM
parameters = {'tfidf__use_idf': (True,False), 'clf__alpha': (1e-2,1e-3)}
svm_clf = GridSearchCV(text_clf, parameters, n_jobs=-1,cv=5, refit=True)
svm_clf = svm_clf.fit(train_all, train_all_target)

#Training accuracy
print("TRAINING ACCURACY")
print("SVM: ",svm_clf.score(train_all, train_all_target))

#-------------------------------------------------------------------------------------------------------------------

print('lr construction')
#We choose to use the lbfgs algorithm which handle l2 penality

text_clf = Pipeline([('tfidf', TfidfTransformer(norm='l2')),('clf',LogisticRegression(solver='lbfgs', penalty='l2',  max_iter= 10, random_state=42))])

#Grid search for parameters of LR
parameters = {'tfidf__use_idf': (True,False), 'clf__fit_intercept': (True,False), 'clf__max_iter': (5,10,100)}
lr_clf = GridSearchCV(text_clf, parameters, n_jobs=-1,cv=5, refit=True)
lr_clf = lr_clf.fit(train_all, train_all_target)

#Training accuracy
print("TRAINING ACCURACY")
print("LR: ",lr_clf.score(train_all, train_all_target))

#-------------------------------------------------------------------------------------------------------------------

print('Naive bayes construction')
text_clf= Pipeline([('tfidf', TfidfTransformer(norm='l2')),
                      ('clf', MultinomialNB())
                      ])
#Grid search for parameters of multiNB
parameters = {'tfidf__use_idf': (True,False), 'clf__alpha': (0.1,1e-2,1e-3), 'clf__fit_prior': (True,False)}
bayes_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=5, refit=True)
bayes_clf = bayes_clf.fit(train_all, train_all_target)


print("TRAINING ACCURACY")
print("Naives Bayes MN: ",bayes_clf.score(train_all, train_all_target))



#-------------------------------------------------------------------------------------------------------------------

def submission(predicted, name):
    idx_test = list(range(1,len(predicted)))
    predicted[predicted==0] = -1
    
    create_csv_submission(idx_test,predicted,name)




#Submission creation
submission(svm_clf.predict(data_test),'./svm.csv')
submission(lr_clf.predict(data_test), './lr_glove.csv')
submission(bayes_clf.predict(data_test),'./nb_glove.csv')

