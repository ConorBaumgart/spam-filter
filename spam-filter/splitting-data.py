from sklearn.model_selection import train_test_split
import numpy as np
import os
from nltk.tokenize import word_tokenize
from pull_data import ham_paths, spam_paths, get_email_content_bulk
from preprocessing import clean_up_pipeline, clean_token_pipeline
from spam_wordcloud import plot_wordcloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Creates a two dimensional list with randomly selected training and testing data from each data source
ham_sample = np.array([train_test_split(o) for o in ham_paths], dtype=object)

ham_train = np.array([])
ham_test = np.array([])
for o in ham_sample:
    ham_train = np.concatenate((ham_train, o[0]), axis = 0)
    ham_test = np.concatenate((ham_test, o[1]), axis = 0)

spam_sample = np.array([train_test_split(o) for o in spam_paths], dtype=object)
spam_train = np.array([])
spam_test = np.array([])
for o in spam_sample:
    spam_train = np.concatenate((spam_train, o[0]), axis = 0)
    spam_test = np.concatenate((spam_test, o[1]), axis = 0)

# Creates array combining ham and spam data, and another array specifying whether each data point is ham or spam.
ham_train_label = [0]*ham_train.shape[0]
spam_train_label = [1]*spam_train.shape[0]
x_train = np.concatenate((ham_train, spam_train))
y_train = np.concatenate((ham_train_label, spam_train_label))

ham_test_label = [0]*ham_test.shape[0]
spam_test_label = [1]*spam_test.shape[0]
x_test = np.concatenate((ham_test,spam_test))
y_test = np.concatenate((ham_test_label,spam_test_label))

# Creates a randomized array of ints and shuffles training and test data
train_shuffle_index = np.random.permutation(np.arange(0, x_train.shape[0]))
test_shuffle_index = np.random.permutation(np.arange(0, x_test.shape[0]))

x_train = x_train[train_shuffle_index]
y_train = y_train[train_shuffle_index]

x_test = x_test[test_shuffle_index]
y_test = y_test[test_shuffle_index]

# Gets email content
x_train = get_email_content_bulk(x_train)
x_test = get_email_content_bulk(x_test)

def remove_null(data, labels):
    not_null_idx = [i for i,o in enumerate(data) if o is not None]
    return np.array(data)[not_null_idx], np.array(labels)[not_null_idx]

x_train, y_train = remove_null(x_train, y_train)
x_test, y_test = remove_null(x_test, y_test)

# Cleans and rearranges data
x_train = [clean_up_pipeline(o) for o in x_train]
x_train = [word_tokenize(o) for o in x_train]
x_train = [clean_token_pipeline(o) for o in x_train]
x_train = [" ".join(o) for o in x_train]

x_test = [clean_up_pipeline(o) for o in x_test]
x_test = [word_tokenize(o) for o in x_test]
x_test = [clean_token_pipeline(o) for o in x_test]
x_test = [" ".join(o) for o in x_test]
# spam_train_index = [i for i,o in enumerate(y_train) if o == 1]
# non_spam_train_index = [i for i,o in enumerate(y_train) if o == 0]

# spam_email = np.array(x_train)[spam_train_index]
# non_spam_email = np.array(x_train)[non_spam_train_index]

# plot_wordcloud(spam_email,title = 'Spam Email')

# plot_wordcloud(non_spam_email,title="Non Spam Email")

x_train = [o.split(" ") for o in x_train]
x_test = [o.split(" ") for o in x_test]
# print(x_train)
vectorizer = TfidfVectorizer()
raw_sentences = [' '.join(o) for o in x_train]
vectorizer.fit(raw_sentences)

def convert_to_feature(raw_tokenize_data):
    raw_sentences = [' '.join(o) for o in raw_tokenize_data]
    return vectorizer.transform(raw_sentences)

x_train_features = convert_to_feature(x_train)
x_test_features = convert_to_feature(x_test)
# print("features")
# print(x_train_features)
# print(x_test_features)

clf = GaussianNB()
print(clf.fit(x_train_features.toarray(), y_train))
print(clf.score(x_test_features.toarray(), y_test))
print(clf.score(x_train_features.toarray(), y_train))

vectorizer = CountVectorizer()
raw_sentences = [' '.join(o) for o in x_train]
vectorizer.fit(raw_sentences)

x_train_features = convert_to_feature(x_train)
x_test_features = convert_to_feature(x_test)
print("features")
print(x_train_features)
print(x_test_features)


clf = GaussianNB()
print(clf.fit(x_train_features.toarray(), y_train))
print(clf.score(x_test_features.toarray(), y_test))
print(clf.score(x_train_features.toarray(), y_train))

lgr = LogisticRegression()
print(lgr.fit(x_train_features.toarray(), y_train))
print(lgr.score(x_test_features.toarray(), y_test))
print(lgr.score(x_train_features.toarray(), y_train))
print(lgr.intercept_)
print(lgr.coef_)