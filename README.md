# TO FIX
- Dans glove_helpers def vectorize_tweet() empty, normal?

# Glove creation

This folder contains all functions used to create the embeddding in 200 dimensions which is used in our models.

# Naives Bayes, SVM, Linear Regression 

We have evaluate theses algorithms on multiple ways to handle the embedding (Glove and the sklearn.feature_extraction.text API)
The dataset we are using can be found in Datasets : test_data.txt, train_neg.txt, train_pos.neg for the one without pre-processing.



## Based on our own GloVe embedding (200dim)

We choose to train our own embedding based on the tweet dataset that was provided. We used the GloVe embedding. We have try differents embeddings size (20,100,200,300) and best results were obtained with 200 dimensions. 

-> The corresponding algoritms can be found in the BasicModelsWithGloVe.py

## Based on the sklearn API text embedding

In the meantime, we used the CountVectorizer() provided by the sklearn.feature_extraction API to transform our text collection in a matrix of tokens.

-> The corresponding algoritms can be found in the BasicModels.py


In both case, we use the l2-norm to compare our vectors. We are using GridSeach to try multiple hyperparameters and cross validation with 5 folds (this one the one with the best result).

# Fast Text -Sentiment analysis

A key concept in Fast Text is to calculate the n-grams of an input sentence and append them to the end of a sentence. Here, we'll use bi-grams. Briefly, a bi-gram is a pair of words/tokens that appear consecutively within a sentence.
In our implementation we used bi-gram. Exemple:"I love Machine Learning": ["I love", "love Machine","Machine Learning"])

The criteriom we use is the BCEWithLogitsLoss that combines a Sigmoid layer and the BCELoss in a single class. 

- Train_full dataset and test dataset have are csv files with 2 columns : 'label' and 'text'
- We use pre-trained GloVE embedding from a twitter dataset with 200dim
- We have constructed two differents neural networks
1) FastText1 : 
input -> Embedding layer -> average pool function -> Linear layer -> output
2) FastText2 with 2 hidden layer: 
input -> Embedding layer -> average pool function -> Linear layer -> average pool func -> Linear layer -> output
We choose to not go futher with the FastText2 as it was giving us really bad prediction.
- Define the Adam optimizer, a BCEWithLogisisLoss as our criterion, train our model on the training dataset, evaluate our model on the validation dataset. 
Stop at the correct epoch before overfitting
- Predict our test data and save it for submission

-> Our Fast Text implementation can be found in fast_text.py

# Convolutional Network -Sentiment analysis

We write down some detail of the implementation: 
- We use max pooling (to extract more important features) and this function handle sequence with different lengths which is important for our convolution filter since their outputs are depending on input size. 
- We use dropout of our convolution filter + linear function at the end
- We use the same optimizer, criterion, training and validating method as before. 
- We have constructed to differents neural networks. Both with the same convolution fiters so 2,3,4,5:
1) CNN1: input -> Embedding layer -> Convolutions Layers -> ReLU activation fonction -> pooled1d activation function for each output of the preceding layer + concatenation -> Dropout -> Linear layer -> Output
2) CNN2: Same as before but with 2 dropouts differents pass through 2 different Linear layers join with 2 different activation function (sigmoid in one and reLU in the second) and then putting both predictions together before the final output. 
The second CNN gaves us poor prediction, so we have choose to forget him for the result description.
- For the prediction, we need to make sure that the imput is at least as long as the largest filter we are using. So don't forget to change the min_len accordingly

The convolution filters act as our n-gramm with n=2,3... of the fast text but we don't need to figure out now which one is the mose interesting because we are testing a lot of them among layers. Theses filters are contains in the convs of each model







