#!/usr/bin/env python
# coding: utf-8


import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from proj2_helpers import create_csv_submission



with open('./Datasets/train_neg.txt', 'r') as f:
    train_neg_sample = f.readlines()
with open('./Datasets/train_pos.txt', 'r') as f:
    train_pos_sample = f.readlines()


train_all = np.concatenate((train_pos_sample, train_neg_sample), axis=0)
train_all_target = np.concatenate((np.ones(len(train_pos_sample)), np.zeros(len(train_neg_sample))),axis=0)

#-------------------------------------------------------------------------------------------------------------------

print('bayes construction')
text_clf= Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())
                      ])
#Grid search for parameters of multiNB
parameters = {'tfidf__use_idf': (True,False), 'clf__alpha': (0.1,1e-2,1e-3), 'clf__fit_prior': (True,False)}
bayes_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=5, refit=True)
bayes_clf = bayes_clf.fit(train_all, train_all_target)


print("TRAINING ACCURACY")
print("Naives Bayes MN: ",bayes_clf.score(train_all, train_all_target))

#-------------------------------------------------------------------------------------------------------------------

print('bayes bernouilli construction')
text_clf= Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', BernoulliNB())
                      ])
#Grid search for parameters of BernouilliNB
parameters = {'tfidf__use_idf': (True,False), 'clf__alpha': (0.1,1e-2,1e-3), 'clf__fit_prior': (True,False)}
bayes_bernoulli_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=5, refit=True)
bayes_bernoulli_clf = bayes_bernoulli_clf.fit(train_all, train_all_target)


print("TRAINING ACCURACY")
print("Bernouilli Bayes MN: ",bayes_bernoulli_clf.score(train_all, train_all_target))

#-------------------------------------------------------------------------------------------------------------------

print('linear svc construction')
text_clf= Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LinearSVC( penalty='l2'))])
#Grid search for parameters of SVC
parameters = {'tfidf__use_idf': (True,False), 'clf__loss': ('hinge','squared_hinge'), 'clf__fit_intercept': (True,False), 'clf__C': np.logspace(-2, 10, 13) }
svc_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=5, refit=True)
svc_clf = svc_clf.fit(train_all, train_all_target)


print("TRAINING ACCURACY")
print('Linear SVC:',svc_clf.score(train_all, train_all_target))

#-------------------------------------------------------------------------------------------------------------------

def submission(predicted, name):
    idx_test = list(range(1,len(predicted)))
    predicted[predicted==0] = -1
    
    create_csv_submission(idx_test,predicted,name)


#Test submissions
data_test = open('./Datasets/test_data.txt', 'r')
submission(bayes_clf.predict(data_test),'./naivesBayes.csv')
submission(bayes_bernoulli_clf.predict(data_test),'./naivesBayesBern.csv')
submission(svc_clf.predict(data_test),'./svm_withoutGloVe.csv')

