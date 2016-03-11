import pandas as pd
import re
from src.utils.utils import *
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


pd.set_option('expand_frame_repr', False)
dtypes = {"name": str, "review": str, "rating": int}
products = pd.read_csv('data/amazon_baby.csv', dtype=dtypes, sep=',', quotechar='"')


def remove_punctuation(text):
    from string import punctuation
    if pd.isnull(text):
        return ''
    return re.sub('[{}]'.format(punctuation), '', text)


def transform_data(prod):
    prod['review_clean'] = prod['review'].apply(remove_punctuation)
    prod = prod[prod['rating'] != 3]
    prod['sentiment'] = prod['rating'].apply(lambda rating: 1 if rating > 3 else -1)
    return prod


def load_train_test_sets(prod):
    train_data_indexes = load_json_list("data/module-2-assignment-train-idx.json")
    train_data = products.iloc[train_data_indexes]
    test_data_indexes = load_json_list("data/module-2-assignment-test-idx.json")
    test_data = products.iloc[test_data_indexes]
    return train_data, test_data


def calc_coefs_fraction(model):
    coefs = sentiment_model.coef_
    positive = len([c for c in coefs[0] if c >= 0])
    negative = len([c for c in coefs[0] if c < 0])
    print("positive coefs: {}, negative coefs: {}".format(positive, negative))
    return positive, negative


def set_up_vectorizer(train_data):
    from os.path import isfile
    path = 'data/serialized/vectorizer.pkl'
    if isfile(path):
        return joblib.load(path)
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    vectorizer.fit_transform(train_data['review_clean'])
    joblib.dump(vectorizer, path)
    return vectorizer


def set_up_model(train_matrix, train_data):
    from os.path import isfile
    path = 'data/serialized/classifier.pkl'
    if isfile(path):
        return joblib.load(path)
    sentiment_model = LogisticRegression()
    sentiment_model.fit(train_matrix, train_data['sentiment'])
    joblib.dump(sentiment_model, path)
    return sentiment_model


def sample_data(test_data):
    sample_test_data = test_data[10:13]
    sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
    scores = sentiment_model.decision_function(sample_test_matrix)
    print(scores)


def best_reviews(test_data, test_matrix, n=20, reverse=False):
    score = sentiment_model.decision_function(test_matrix)
    indexed_score = zip(score, range(len(score)))
    indexes = [index for (value, index) in sorted(indexed_score, reverse=not reverse)[:20]]
    reves = test_data[['name', 'review']].iloc[indexes]
    for rev in reves.iterrows():
        print(rev[1]['name'], '\n')
    print(indexes)


def sentiment_model_acc(prediction, test_labels):
    from sklearn.metrics import accuracy_score
    print(accuracy_score(prediction, test_labels))


products = transform_data(products)
train_data, test_data = load_train_test_sets(products)
vectorizer = set_up_vectorizer(train_data)
train_matrix = vectorizer.transform(train_data['review_clean'])
sentiment_model = set_up_model(train_matrix, train_data)
calc_coefs_fraction(sentiment_model)
sample_data(test_data)
test_matrix = vectorizer.transform(test_data['review_clean'])
test_labels = test_data['sentiment']
prediction = sentiment_model.predict(test_matrix)
print(prediction[:200])
print(test_labels[:200])
sentiment_model_acc(prediction, test_labels)


# best_reviews(test_data, test_matrix)
# best_reviews(test_data, test_matrix, reverse=True)




# probabilities = sentiment_model.predict_proba(sample_test_matrix)
# print("probas", probabilities)
#
# from scipy.stats import logistic
# print(logistic.cdf(scores[0]))

# print(sentiment_model.predict(sample_test_matrix))
