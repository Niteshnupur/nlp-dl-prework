# --------------
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix





# Code starts here

# load data

# Load data using path variable using .read_csv() method of pandas. Save it as news

news = pd.read_csv(path)


# subset data

# All the columns are not relevant and you will be going forward with TITLE (title of resource) and CATEGORY (class label). Select only these two features and store it as news

news = news[["TITLE" , "CATEGORY"]]




# distribution of classes

# Now, get the class distribution of class lables. You can get it using .value_counts() method on CATEGORY column of news dataframe. Store it inside dist variable

dist = news["CATEGORY"].value_counts

# display class distribution

print(dist)

# display data

print(news.head())

# Code ends here


# --------------
# Preprocess data and split into training and test sets
# In the previous task you selected the required subset of data and observed the distribution of labels. Now time to preprocess text data by doing the following steps on the TITLE column -

# Retaining only alphabets (Using regular expressions)
# Removing stopwords (Using nltk library)
# Splitting into train and test sets (Uisng scikit-learn library)​


# Code starts here

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix 
import nltk
nltk.download('stopwords')


# Initialize stopwords as stop with set(stopwords.words('english'))

stop = set(stopwords.words('english'))


# retain only alphabets

# To retain only alphabets for every instance, use a lambda function in combination with .apply() method that does so. The function that you will be applying to every instance (supposing the instance is row) will be re.sub("[^a-zA-Z]", " ",x). Remember this operation should be carried out on TITLE column only

news['TITLE'] = news['TITLE'].apply(lambda x : re.sub('[^a-zA-Z]', ' ',x))



# convert to lowercase and tokenize

# Next use lambda function and .apply() method to first convert the instances to lowercase (using .lower()) and then tokenize (using .split()). Remember this operation should be carried out on TITLE column on

news['TITLE'] = news['TITLE'].apply(lambda x:x.lower().split())



# remove stopwords

# Now time to remove stopwords from every instance. Again using a combination of lambda function and .apply() method retain only words which are in that instance but not in stop. You can take the help of a list comprehension to achieve it. For ex: [i for i in x if i not in y]. Remember this operation should be carried out on TITLE column only


news['TITLE'] = news['TITLE'].apply(lambda x:[i for i in x if i not in stop])


# join list elements

# The steps mentioned above gives a list for every instance across TITLE column. Join the list elements into a single sentence using ' '.join() method of lists. Use both lambda function and .apply() method for it.

news['TITLE'] = news['TITLE'].apply(lambda x: ' '.join(x))


# split into training and test sets

# Finally split into train and test using train_test_split function where feature is news["TITLE"], target is news["CATEGORY"], test size is 20% and random state is 3. Save the resultant variables as X_train, X_test, Y_train and Y_test

X_train, X_test, Y_train , Y_test = train_test_split(news["TITLE"], news["CATEGORY"], test_size = 0.2,random_state=3)






# Code ends here


# --------------
# Vectorize with Bag-of-words and TF-IDF approach
# After cleaning data its time to vectorize data so that it can be fed into an ML algorithm. You will be doing it with two approaches: Bag-of-words and TF-IDF.


# Code starts here

# Initialize Bag-of-words vectorizer using CountVectorizer() and TF-IDF vectorizer using TfidfVectorizer(ngram_range=(1,3)). Save them as count_vectorizer and tfidf_vectorizer respectively

# initialize count vectorizer
count_vectorizer = CountVectorizer()

# initialize tfidf vectorizer

tfidf_vectorizer  = TfidfVectorizer(ngram_range=(1,3))


# Next thing to do is fit each vectorizer on training and test features with text data and transform them to vectors.

# fit and transform with count vectorizer

X_train_count = count_vectorizer.fit_transform(X_train)

X_test_count = count_vectorizer.transform(X_test)



# Similarly repeat the previous two steps with tfidf_vectorizer and save the transformed training feature as X_train_tfidf and transformed test feature as X_test_tfidf

# fit and transform with tfidf vectorizer

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Code ends here


# --------------
# Predicting with Multinomial Naive Bayes
# Multinomial Naive Bayes is an algorithm that can be used for the purpose of multi-class classification. You will be using it to train and test it on both the versions i.e. Bag-of-words and TF-IDF ones and then checking the accuracy on both of them


# Code starts here

# First initialize two Multinomial Naive Bayes classifiers with MultinomialNB() and save them as nb_1 and nb_2. The reason for initializing two classifiers is because you will be training and testing on both Bag-of-words and TF-IDF transformed training data


# initialize multinomial naive bayes

nb_1 = MultinomialNB()

nb_2 = MultinomialNB()


# fit on count vectorizer training data

# Fit nb_1 on X_train_count and Y_train using .fit() method

nb_1.fit(X_train_count , Y_train)



# fit on tfidf vectorizer training data

# Fit nb_2 on X_train_tfidf and Y_train using .fit() method

nb_2.fit(X_train_tfidf,Y_train)


# accuracy with count vectorizer

# Find the accuracy with Bag-of-words approach using accuracy_score(nb_1.predict(X_test_count), Y_test) and save it as acc_count_nb

acc_count_nb =accuracy_score(nb_1.predict(X_test_count),Y_test)


# Similarly find the accuracy for the TF-IDF approach (only difference is the classifer is nb_2) and save it as acc_tfidf_nb

# # accuracy with tfidf vectorizer

acc_tfidf_nb = accuracy_score(nb_2.predict(X_test_tfidf),Y_test)



# display accuracies
# Print out acc_count_nb and acc_tfidf_nb to check which version performs better for with Multinomial Naive Bayes as classifer


print(acc_count_nb)

print(acc_tfidf_nb)



# Code ends here


# --------------
# Predicting with Logistic Regression
# Logistic Regression can be used for binary classification but when combined with OneVsRest classifer, it can perform multiclass classification as well. You will be using one such algorithm to train and test it on both the versions i.e. Bag-of-words and TF-IDF ones and then checking the accuracy on both of them


import warnings
warnings.filterwarnings('ignore')


# initialize logistic regression

# First initialize two classifiers with OneVsRestClassifier(LogisticRegression(random_state=10)) and save them as logreg_1 and logreg_2. The reason for initializing two classifiers is because you will be training and testing on both Bag-of-words and TF-IDF transformed training data

logreg_1 = OneVsRestClassifier(LogisticRegression(random_state=10))

logreg_2 = OneVsRestClassifier(LogisticRegression(random_state=10))


# Fit logreg_1 on X_train_count and Y_train using .fit() method
# fit on count vectorizer training data

logreg_1.fit(X_train_count,Y_train)


#Fit logreg_2 on X_train_tfidf and Y_train using .fit() method
# fit on tfidf vectorizer training data

logreg_2.fit(X_train_tfidf,Y_train)


# Find the accuracy with Bag-of-words approach using accuracy_score(logreg_1.predict(X_test_count), Y_test) and save it as acc_count_logreg

# accuracy with count vectorizer

acc_count_logreg = accuracy_score(logreg_1.predict(X_test_count),Y_test)


# Similarly find the accuracy for the TF-IDF approach (only difference is the classifer is logreg_2) and save it as acc_tfidf_logreg

# accuracy with tfidf vectorizer

acc_tfidf_logreg = accuracy_score(logreg_2.predict(X_test_tfidf), Y_test)



# Print out acc_count_logreg and acc_tfidf_logreg to check which version performs better for with Multinomial Naive Bayes as classife



# display accuracies

print(acc_count_logreg)

print(acc_tfidf_logreg)





# Code ends here


