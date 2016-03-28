import pandas as pd
from src.utils.utils import *
from src.utils.config import project_root, pandas_setup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


def transform_data(prod):
    prod['review_clean'] = prod['review'].apply(remove_punctuation)
    prod = prod[prod['rating'] != 3]
    prod['sentiment'] = prod['rating'].apply(lambda rating: 1 if rating > 3 else -1)
    return prod


def calc_coefs_fraction(model):
    coefs = model.coef_
    positive = len([c for c in coefs[0] if c >= 0])
    negative = len([c for c in coefs[0] if c < 0])
    print('--------coefs----------')
    print("positive coefs: {}, negative coefs: {}".format(positive, negative))
    return positive, negative


def set_up_vectorizer(train_data, words=None):
    from os.path import isfile
    simple = ''
    if words:
        simple = 'simple'
    path = project_root('data/serialized/{}vectorizer.pkl'.format(simple))
    if isfile(path):
        return joblib.load(path)
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', vocabulary=words)
    vectorizer.fit_transform(train_data['review_clean'])
    joblib.dump(vectorizer, path)
    return vectorizer


def set_up_model(train_matrix, train_data, simple=''):
    from os.path import isfile
    path = project_root('data/serialized/{}classifier.pkl'.format(simple))
    if isfile(path):
        return joblib.load(path)
    sentiment_model = LogisticRegression()
    sentiment_model.fit(train_matrix, train_data['sentiment'])
    joblib.dump(sentiment_model, path)
    return sentiment_model


def sample_data(vectorizer, sentiment_model, test_data):
    sample_test_data = test_data[10:13]
    sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
    scores = sentiment_model.decision_function(sample_test_matrix)
    print(scores)


def best_reviews(test_data, sentiment_model, test_matrix, n=20, reverse=False):
    score = sentiment_model.decision_function(test_matrix)
    indexed_score = zip(score, range(len(score)))
    indexes = [index for (value, index) in sorted(indexed_score, reverse=not reverse)[:n]]
    reves = test_data[['name', 'review']].iloc[indexes]
    best = 'best {}'.format(n)
    if reverse:
        best = 'worst {}'.format(n)
    print('-------{}-------'.format(best))
    for rev in reves.iterrows():
        print(rev[1]['name'])
    # print(indexes)


def sentiment_model_acc(prediction, test_labels, on):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(prediction, test_labels)
    print('-------accuracy {}: {}-------'.format(on, accuracy))
    return accuracy


def check_coef_weight(vectorizer, model, word):
    vectorized = vectorizer.transform([word])
    return model.decision_function(vectorized) - model.intercept_


def model_features(vectorizer, model, words):
    print('------model features----------')
    for word in words:
        print(word, check_coef_weight(vectorizer, model, word))
    calc_coefs_fraction(model)


def asses_sentiment_model(train_data, test_data, words):
    vectorizer = set_up_vectorizer(train_data)
    train_matrix = vectorizer.transform(train_data['review_clean'])
    sentiment_model = set_up_model(train_matrix, train_data)
    calc_coefs_fraction(sentiment_model)
    sample_data(vectorizer, sentiment_model, test_data)
    test_matrix = vectorizer.transform(test_data['review_clean'])
    test_labels = test_data['sentiment']
    train_labels = train_data['sentiment']
    prediction = sentiment_model.predict(test_matrix)
    prediction_train = sentiment_model.predict(train_matrix)
    sentiment_model_acc(prediction, test_labels, 'sentiment on test')
    sentiment_model_acc(prediction_train, train_labels, 'sentiment on train')
    best_reviews(test_data, sentiment_model, test_matrix)
    best_reviews(test_data, sentiment_model, test_matrix, reverse=True)
    model_features(vectorizer, sentiment_model, words)


def majority_classifier(train_data, test_data):
    print('--------majority model---------')
    train = train_data['sentiment']
    positives = len([c for c in train if c >= 0])
    clazz = -1
    if len(train) < 2 * positives:
        clazz = 1
    from sklearn.metrics import accuracy_score
    test = test_data['sentiment']
    prediction = [clazz for i in range(len(test))]
    accuracy = accuracy_score(test, prediction)
    print('accuracy:', accuracy)


def asses_simple_model(train_data, test_data, significant_words):
    vectorizer_word_subset = set_up_vectorizer(train_data, significant_words)
    train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
    test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])
    simple_sentiment_model = set_up_model(train_matrix_word_subset, train_data, 'simple')
    prediction = simple_sentiment_model.predict(test_matrix_word_subset)
    prediction_train = simple_sentiment_model.predict(train_matrix_word_subset)
    test_labels = test_data['sentiment']
    train_labels = train_data['sentiment']
    sentiment_model_acc(prediction, test_labels, 'simple sentiment on test')
    sentiment_model_acc(prediction_train, train_labels, 'simple sentiment on train')
    model_features(vectorizer_word_subset, simple_sentiment_model, significant_words)


def main():
    pandas_setup()
    dtypes = {"name": str, "review": str, "rating": int}
    products = pd.read_csv(project_root('data/amazon_baby.csv'), dtype=dtypes, sep=',', quotechar='"')

    significant_words = {'love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
                         'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
                         'work', 'product', 'money', 'would', 'return'}

    products = transform_data(products)
    train_data, test_data = load_train_test_sets(products,
                                                 "module-2-assignment-train-idx.json",
                                                 "module-2-assignment-test-idx.json")
    asses_sentiment_model(train_data, test_data, significant_words)
    asses_simple_model(train_data, test_data, significant_words)
    majority_classifier(train_data, test_data)


if __name__ == '__main__':
    main()