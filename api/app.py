import os
import glob
import email
import string
import re

from flask import Flask, request
from flask_cors import CORS, cross_origin
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

nltk.download('punkt')
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
clf = LogisticRegression()


@app.route('/spamorham', methods=['POST'])
@cross_origin()
def spamorham():
    data = request.get_json()
    results = user_message_analysis(data["message"])
    return results


def main():
    path = '../../temp/spamham/'

    easy_ham_paths = glob.glob(path+'easy_ham/*')
    easy_ham_2_paths = glob.glob(path+'easy_ham_2/*')
    hard_ham_paths = glob.glob(path+'hard_ham/*')
    spam_paths = glob.glob(path+'spam/*')
    spam_2_paths = glob.glob(path+'spam_2/*')

    def get_email_content(email_path):
        file = open(email_path,encoding='latin1')
        try:
            msg = email.message_from_file(file)
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    return part.get_payload() # prints the raw text
        except Exception as e:
            print(e)    
            
    def get_email_content_bulk(email_paths):
        email_contents = [get_email_content(o) for o in email_paths]
        return email_contents

    ham_path = [
        easy_ham_paths,
        easy_ham_2_paths,
        hard_ham_paths
    ]

    spam_path = [
        spam_paths,
        spam_2_paths
    ]
    
    ham_sample = np.array([train_test_split(o) for o in ham_path], dtype=object)

    ham_train = np.array([])
    ham_test = np.array([])
    for o in ham_sample:
        ham_train = np.concatenate((ham_train,o[0]),axis=0)
        ham_test = np.concatenate((ham_test,o[1]),axis=0)
    
    spam_sample = np.array([train_test_split(o) for o in spam_path], dtype=object)
    spam_train = np.array([])
    spam_test = np.array([])
    for o in spam_sample:
        spam_train = np.concatenate((spam_train,o[0]),axis=0)
        spam_test = np.concatenate((spam_test,o[1]),axis=0)

    ham_train_label = [0]*ham_train.shape[0]
    spam_train_label = [1]*spam_train.shape[0]
    x_train = np.concatenate((ham_train,spam_train))
    y_train = np.concatenate((ham_train_label,spam_train_label))

    ham_test_label = [0]*ham_test.shape[0]
    spam_test_label = [1]*spam_test.shape[0]
    x_test = np.concatenate((ham_test,spam_test))
    y_test = np.concatenate((ham_test_label,spam_test_label))

    train_shuffle_index = np.random.permutation(np.arange(0,x_train.shape[0]))
    test_shuffle_index = np.random.permutation(np.arange(0,x_test.shape[0]))

    x_train = x_train[train_shuffle_index]
    y_train = y_train[train_shuffle_index]

    x_test = x_test[test_shuffle_index]
    y_test = y_test[test_shuffle_index]

    x_train = get_email_content_bulk(x_train)   # long run time
    x_test = get_email_content_bulk(x_test) # long run time

    def remove_null(datas,labels):
        not_null_idx = [i for i,o in enumerate(datas) if o is not None]
        return np.array(datas)[not_null_idx],np.array(labels)[not_null_idx]
    
    x_train,y_train = remove_null(x_train,y_train)
    x_test,y_test = remove_null(x_test,y_test)

    x_train = [clean_up_pipeline(o) for o in x_train] # remove whitespace, etc., lowercase
    x_test = [clean_up_pipeline(o) for o in x_test]

    x_train = [word_tokenize(o) for o in x_train]   # long run time - split sentence to array
    x_test = [word_tokenize(o) for o in x_test] # long run time

    x_train = [clean_token_pipeline(o) for o in x_train] # remove stop words, lemmatize
    x_test = [clean_token_pipeline(o) for o in x_test]
    
    raw_sentences = [' '.join(o) for o in x_train]
    vectorizer.fit(raw_sentences)

    x_train_features = convert_to_feature(x_train)
    x_test_features = convert_to_feature(x_test)

    clf.fit(x_train_features.toarray(),y_train) # long
    print("Testing Score", clf.score(x_test_features.toarray(),y_test))
    print("Training Score", clf.score(x_train_features.toarray(),y_train))


def user_message_analysis(message):
    x_new_features = clean_message(message)
    prediction = clf.predict(x_new_features.toarray())
    probability = clf.predict_proba(x_new_features.toarray())
    spam = "Yes" if prediction == 1 else "No"
    yesSpam = round(probability[0][1] * 100, 1)
    noSpam = round(probability[0][0] * 100, 1)
    results = {"spam": spam, "chance_yes": f"{yesSpam}%", "chance_no": f"{noSpam}%"}
    return results


def clean_message(message):
    x_new = [message]
    x_new = [clean_up_pipeline(o) for o in x_new] # remove whitespace, etc., lowercase
    x_new = [word_tokenize(o) for o in x_new]   # long run time - split sentence to array
    x_new = [clean_token_pipeline(o) for o in x_new] # remove stop words, lemmatize
    x_new_features = convert_to_feature(x_new)
    clf.predict(x_new_features.toarray())
    clf.predict_proba(x_new_features.toarray())
    return x_new_features


def remove_hyperlink(word):
    return  re.sub(r"http\S+", "", word)

def to_lower(word):
    result = word.lower()
    return result

def remove_number(word):
    result = re.sub(r'\d+', '', word)
    return result

def remove_punctuation(word):
    result = word.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result

def remove_whitespace(word):
    result = word.strip()
    return result

def replace_newline(word):
    return word.replace('\n','')

def clean_up_pipeline(sentence):
    cleaning_utils = [remove_hyperlink,
                      replace_newline,
                      to_lower,
                      remove_number,
                      remove_punctuation,remove_whitespace]
    for o in cleaning_utils:
        sentence = o(sentence)
    return sentence

def remove_stop_words(words):
    result = [i for i in words if i not in ENGLISH_STOP_WORDS]
    return result

def word_stemmer(words):
    return [stemmer.stem(o) for o in words]

def word_lemmatizer(words):
    return [lemmatizer.lemmatize(o) for o in words]

def clean_token_pipeline(words):
    cleaning_utils = [remove_stop_words,word_lemmatizer]
    for o in cleaning_utils:
        words = o(words)
    return words

def convert_to_feature(raw_tokenize_data):
    raw_sentences = [' '.join(o) for o in raw_tokenize_data]
    return vectorizer.transform(raw_sentences)

main()
