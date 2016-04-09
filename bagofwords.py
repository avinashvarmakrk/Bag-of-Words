import pandas as pd
train = pd.read_csv("labeledTrainData.tsv",header = 0,delimiter = "\t", quoting= 3)
train.shape
train.columns.values
print train["review"][0]

from bs4 import BeautifulSoup
example1 = BeautifulSoup(train["review"][0])
print example1.get_text()

import re
letters_only = re.sub("[^a-zA-Z]"," ",example1.get_text())
print letters_only

lower_case = letters_only.lower()
words = lower_case.split()

import nltk
nltk.download("all")

from nltk.corpus import stopwords
from nltk import PorterStemmer
print stopwords.words("english")

words = [w for w in words if not w in stopwords.words("english")]
print words

def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))  
    stops = PorterStemmer().stem_word("stops")                
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops] 
      
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 
    
num_reviews = train["review"].size  
clean_train_reviews = []
for i in xrange(0,num_reviews):
    clean_train_reviews.append( review_to_words( train["review"][i] ) )
    
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features= 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
print train_data_features.shape

vocab = vectorizer.get_feature_names()
print vocab

import numpy as np
dist = np.sum(train_data_features,axis = 0)
for tag, count in zip(vocab,dist):
    print count, tag
    
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=1000)
forest = forest.fit(train_data_features,train["sentiment"])


test = pd.read_csv("testData.tsv",header = 0, delimiter = "\t", quoting = 3)
print test.shape
num_reviews = len(test["review"])
clean_test_reviews = []
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )
    
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
result = forest.predict(test_data_features)
output = pd.DataFrame(data = {"id":test["id"],"sentiment":result})
output.to_csv("Bag_of_words_model.csv",index = False,quoting = 3)