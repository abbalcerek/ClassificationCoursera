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

products = transform_data(products)
train_data, test_data = load_train_test_sets(products)

# vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
vectorizer = joblib.load("data/serialized/vectorizer.pkl")
# train_matrix = vectorizer.fit_transform(train_data['review_clean'])
train_matrix = vectorizer.transform(train_data['review_clean'])



# sentiment_model = LogisticRegression()
# sentiment_model.fit(train_matrix, train_data['sentiment'])
sentiment_model = joblib.load("data/serialized/classifier.pkl")


calc_coefs_fraction(sentiment_model)

sample_test_data = test_data[10:13]

joblib.dump(sentiment_model, 'data/serialized/classifier.pkl')
joblib.dump(vectorizer, 'data/serialized/vectorizer.pkl')

print(sample_test_data.iloc[0]['review'])
print(sample_test_data.iloc[1]['review'])
print(sample_test_data.iloc[2]['review'])

print(sample_test_data)

sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)
print(scores)

probabilities = sentiment_model.predict_proba(sample_test_matrix)
print("probas", probabilities)

from scipy.stats import logistic
print(logistic.cdf(scores[0]))


test_matrix = vectorizer.transform(test_data['review_clean'])
labels = test_data['rating']

prediction = sentiment_model.predict(test_matrix)
print(prediction[0:100])

print(sentiment_model.predict(sample_test_matrix))
