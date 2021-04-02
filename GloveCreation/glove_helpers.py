# importing libraries
import numpy as np
import pandas as pd

def features_creation(file_path, output_name, words_vect, vocab, common_words=False, pond=1, verbose=False):
    """creates a file that contains the vectorized tweets. 
    The file can then be transformed into a pandas datafame using pd.read_csv(output, index_col=0)"""
    
    # loading common words
    if(common_words):
        common_words_list = np.loadtxt('./common_words.txt', dtype='str', delimiter='\n')
    
    # loading vectorized words and vocabulary
    dim = words_vect.shape[1] # number of features
    
    tweet_text = np.loadtxt(file_path, dtype='str', delimiter='\n')
    tweet_word_lists = np.array([])
    
    # transfoming 8tweets into an array of words (non vectorized)
    for t in tweet_text:
        tweet_word_lists = np.append(tweet_word_lists, np.char.split(t))
        
    # compute tweets features by averaging the vectorized words
        tweet_vectorized = np.array([])
    
    # iterating over all positive tweets
    print("Vectorizing {p} tweets...".format(p=tweet_word_lists.shape[0]))
    i = 1
    for t in tweet_word_lists:
        words = np.zeros(dim) # array that will store the vectorized words

        # iterating over the words
        for w in t: 
            indice = vocab.get(w) 

            # testing if word is in the dictionnary
            if indice is None: 
                word_vect = np.zeros(dim)
            else:
                word_vect = words_vect[indice] # get the vector corresponding to the word
                
            # if pond_common_words, then then most common weight have a higher weight. 
            if common_words:
                if(w in common_words_list):
                    word_vect = word_vect * pond
                
            words = np.vstack((words, word_vect))

        #averaging
        words_avg = words.mean(axis=0) 

        # appending to array of vectorized tweets...
        tweet_vectorized = np.append(tweet_vectorized, words_avg, axis=0) 
        i =  i + 1
        if(i % 1000 == 0 and verbose):
            print("{t} tweets processed".format(t=i))

    # ...and then reshaping
    tweet_vectorized = np.reshape(tweet_vectorized, newshape=(-1, dim))
    
    # writing features of positive and negative tweets into csv files
    pd.DataFrame(tweet_vectorized).to_csv(output_name)
    print('done.')
    
    
    
def features_creation_opt(file_path, output_name, words_vect, vocab, common_words=False, pond=1, verbose=False):
    """creates a file that contains the vectorized tweets. 
    The file can then be transformed into a pandas datafame using pd.read_csv(output, index_col=0)"""
    
    # loading common words
    if(common_words):
        common_words_list = np.loadtxt('./common_words.txt', dtype='str', delimiter='\n')
    
    # loading vectorized words and vocabulary
    dim = words_vect.shape[1] # number of features
    
    print('Loading the tweets...')
    
    tweet_text = np.loadtxt(file_path, dtype='str', delimiter='\n')
    
    print('Transforming the tweets into a list of words...')
    
    # transfoming 8tweets into an array of words (non vectorized)
    tweet_word_lists = np.array([t.split(' ') for t in tweet_text])
    
    print("Vectorizing {p} tweets...".format(p=tweet_word_lists.shape[0]))
        
    # compute tweets features by averaging the vectorized words
    tweet_vectorized = np.zeros((tweet_word_lists.shape[0], dim))
    
    # iterating over all positive tweets
    i = 1
    for t in range(0, tweet_word_lists.shape[0]):
        words = np.zeros(dim) # array that will store the vectorized words

        # iterating over the words
        for w in tweet_word_lists.item(t): 
            indice = vocab.get(w) 

            # testing if word is in the dictionnary
            if indice is None: 
                word_vect = np.zeros(dim)
            else:
                word_vect = words_vect[indice] # get the vector corresponding to the word
                
            # if pond_common_words, then then most common weight have a higher weight. 
            if common_words:
                if(w in common_words_list):
                    word_vect = word_vect * pond
                
            words = np.vstack((words, word_vect))

        #averaging
        words_avg = words.mean(axis=0) 

        # appending to array of vectorized tweets...
        tweet_vectorized[t] = words_avg
        i =  i + 1
        if(i % 1000 == 0 and verbose):
            print("{t} tweets processed".format(t=i))

    print('Writing the vectorized tweets in the output file...')
    
    # writing features of positive and negative tweets into csv files
    pd.DataFrame(tweet_vectorized).to_csv(output_name)
    print('done.')
    
def features_creation_pretrained(pretrained_data_path, file_path, output_name):
    
    emb = pd.read_csv('pretrained_data_path', header = None, sep=' ')
    emb.set_index(0, inplace=True)
    indices = emb.index.values
    dim = emb.shape[0]

    tweets_text = np.loadtxt(file_path, dtype='str', delimiter='\n')
    tweet_word_lists = np.array([t.split(' ') for t in tweets_text])

    tweet_vectorized = np.zeros((tweet_word_lists.shape[0], dim))
    
    i = 1    
    # iterating over all positive tweets
    for t in range(0, tweet_word_lists.shape[0]):
        words = np.zeros(dim) # array that will store the vectorized words

        # iterating over the words
        for w in tweet_word_lists.item(t): 

            # testing if word is in the dictionnary
            if w in indices: 
                word_vect = emb.loc[w].values # get the vector corresponding to the word
            else:
                word_vect = np.zeros(dim)

            words = np.vstack((words, word_vect))

        #averaging
        words_avg = words.mean(axis=0) 

        # appending to array of vectorized tweets...
        tweet_vectorized[t] = words_avg
        i =  i + 1
        if(i % 100 == 0):
            print("{t} tweets processed".format(t=i))

    print('Writing the vectorized tweets in the output file...')

    # writing features of positive and negative tweets into csv files
    pd.DataFrame(tweet_vectorized).to_csv(output_name)
    print('done.')
    
def vectorize_tweet()
