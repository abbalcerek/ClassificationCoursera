import pandas as pd
import re
from src.utils.utils import *
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

pd.set_option('expand_frame_repr', False)
dtypes = {"name": str, "review": str, "rating": int}
products = pd.read_csv('data/amazon_baby.csv', dtype=dtypes, sep=',', quotechar='"')


def remove_punctuation(text):
    from string import punctuation
    if pd.isnull(text):
        return ''
    return re.sub('[{}]'.format(punctuation), '', text)

products['review_clean'] = products['review'].apply(remove_punctuation)
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating: 1 if rating > 3 else -1)

train_data_indexes = load_json_list("data/module-2-assignment-train-idx.json")
train_data = products.iloc[train_data_indexes]

test_data_indexes = load_json_list("data/module-2-assignment-test-idx.json")
test_data = products.iloc[train_data_indexes]

print(train_data)


vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
print(train_matrix)
print(train_matrix.shape)

sentiment_model = LogisticRegression()
sentiment_model.fit(train_matrix, train_data['sentiment'])

coefs = sentiment_model.coef_
print(coefs.shape)
positive = len([c for c in coefs[0] if c >= 0])
negative = len([c for c in coefs[0] if c < 0])

print("positive coefs: {}, negative coefs: {}".format(positive, negative))