# 2020_Antisemitism_Hackathon  
## Project Summary
This is the source code for the IU Antisemitism Datathon and Hackathon 2020. The final project classifier is an LSTM Keras network using word embeddings and several hidden layers. We experimented using several different classification methods before concluding that the LSTM network gave superior performance, with a test F1 score of 79%.

# Methods Explored:
## 1) Ngrams Model with Tf-Idf vectorization
Used NLTK to preprocess text of each tweet along with the user's profile description. The tweet text concatenated with the user's profile description was stored under the feature column "total_text". After setting text to lower case, stemming, lemmatizing, and removing stop words, there were ~550 unique words. Used NLTK's frequency distribution to find unigrams, bigrams, and trigrams that appeared more frequently(10 times more) in antisemitic tweets than clean tweets(or vice-versa). Those most significant ngrams were used for the next phase of the model. We used sklearn to extract the Term Frequency-Inverse Document Frequency(Tf-Idf) for each ngram. This method gives higher scores to ngrams that appear less frequently in the corpus as a whole but more frequently in an individual tweet, thus giving more weight to terms significant to any given tweet. The vectors of the Tfidf frequency of these ngrams served as inputs for a Naive Bayes Classifier.

## 2) Spacy Text Classifiers
Vectorized the tweet data using the same ngram vocabularly determined in the first NLTK model. The data was vectorized using only a CountVectorizer instead of the tf-idf method of determining ngram frequency. Used sklearn's Pipeline to easily test this method of text preparation on a variety of different models. Tested models including LogisticRegression, Naive Bayes(MultinomialNB), Support Vector Classifier(SVC), RandomForestClassifier, AdaBoostClassifier, and RandomForest Classifier. Finally, these algorithms were combined into a VoterClassifier where each subclass was given a single vote. The AdaBoostClassifier and the VotingClassifer had the highest accuracy and f1score of all of the models tested.

## 3) LSTM classifier
Used Keras Tokenizer, which converted all text to lowercase and stripped punctuation but otherwise skipped text preprocessing used in earlier models. Vectorized data is fed into deep network with several hidden layers, including an LSTM layer, a dropout layer, and several dense sigmoid layers. Despite the decreased amount of text preprocessing, this network outperformed all other models and was ultimately chosen to classify the data.

## 4) Classifier Combination
We hypothesized that the LSTM networks already strong predictions could be augmented through the incorporation of prediction by the other classifiers. We engineered feature columns representing the predictions of the TfidfClassifier, VoterClassifier, AdaBoostClassifier, and LSTM network. These predictions were used as input for a Support Vector Classifier(SVC). However, this ensemble classifier performed worse when measured by the accuracy and f1scores than the LSTM network alone. This suggests that the LSTM network already encompasses all of the information gleaned by the previous text classifiers, and thus was unaided by their input. Consequently, we chose to use solely the LSTM network.

## Final Result
The LSTM network outputs a probability between 0 and 1 that the tweet is antisemitic. We found that feeding this input into the SVC, which outputted a binary prediction, obtained greater accuracy than setting an arbitrary threshold such as 0.5 and splitting the LSTM probability into a binary output in that manner. The SVC determines the optimal threshold. Thus, the final classifier is an SVC that outputs a binary probability given single input of the probability determined by the LSTM network.

The "Parent Classifier" outputs the binary prediction given the probability that the tweet is antisemitic according to the LSTM Classifier.
The "Parent Classifier" is an SVM that determines the best probability threshold for classification.

# To run frequency_analysis.ipynb and replicate results
1) Open the file in Google Colaboratory.
2) Upload training data and testing data.
Click the folder icon to the left of the screen and upload the two files, then change the two variable names below to the names of the two data files.
3) Run All Cells
4) View the output of the last cell in the notebook to see the F1 Score of the classifier.
