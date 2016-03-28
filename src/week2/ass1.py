from src.utils.utils import *
from src.utils.config import *
import pandas as pd
import numpy as np


IMPORTANT_WORDS = load_json_list(project_root("data/important_words.json"))


def explore_data(products):
    print('--------explore data--------')
    print('---10 first names---')
    print(products[:10]['name'])
    positive = [value for value in products['sentiment'] if value == 1]
    negative = [value for value in products['sentiment'] if value == -1]
    print('positive:', len(positive), ', negative:', len(negative))
    print('perfect count:', len([True for i in products['perfect'] if i > 0]))


def clean_data(products):
    products['review'] = products['review'].fillna('')
    revs = products['review']
    products['review_clean'] = products['review']\
        .apply(remove_punctuation)

    for rev in revs:
        if not isinstance(rev, str):
            print(rev)


def add_word_counts(products):
    for word in IMPORTANT_WORDS:
        products[word] = products['review_clean']\
            .apply(lambda s : s.split().count(word))


def set_up_products():
    import os
    from os.path import isfile, join
    from sklearn.externals import joblib
    dir = project_root('data/serialized2/products')
    path = join(dir, "products.pkl")
    if isfile(path):
        return joblib.load(path)
    products = pd.read_csv(project_root('data/amazon_baby_subset.csv'))
    clean_data(products)
    add_word_counts(products)
    if not os.path.exists(dir): os.makedirs(dir)
    joblib.dump(products, path)
    return products


def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    print(IMPORTANT_WORDS[0:10])
    print(features_frame.columns.values[0:11])
    feature_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return feature_matrix, label_array


def predict_probability(feature_matrix, coefficients):
    score = feature_matrix.dot(coefficients)
    return vSigmoid(score)


def feature_derivative(errors, feature):
    derivative = feature.transpose().dot(errors)
    return derivative


def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment + 1) / 2
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores)))
    return lp


def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in range(max_iter):
        predictions = predict_probability(feature_matrix, coefficients)
        indicator = (sentiment + 1) / 2

        errors = indicator - predictions
        derivative = feature_derivative(errors, feature_matrix)
        coefficients += step_size * derivative

        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print('iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp))
    return coefficients


def run_lg(feature_matrix, labels):
    initial_coefs = np.zeros(feature_matrix.shape[1])
    return logistic_regression(feature_matrix, labels, initial_coefs, 1e-7, 301)


def predict(coefs, feature_matrix):
    product = feature_matrix.dot(coefs)
    return np.vectorize(lambda scalar: 1 if scalar > 0. else -1)(product)


def evaluate(coefs, labels, prediction):
    from sklearn.metrics import accuracy_score
    print('positive review count: {}'.format(np.count_nonzero(prediction + np.ones(prediction.shape))))
    print('accuracy: {}'.format(accuracy_score(labels, prediction)))
    important_words(coefs, IMPORTANT_WORDS, True)
    important_words(coefs, IMPORTANT_WORDS, False)


def important_words(coefs, words, best=True, number=10):
    print('------------important words ---------------')
    sorted_pairs = sorted(zip(coefs[1:], words), reverse=best)
    print(sorted_pairs)
    for (coef, word) in sorted_pairs[0: number]:
        print('weight of word {}: {}'.format(word, coef))


def main():
    products = set_up_products()
    explore_data(products)
    feature_matrix, label_array \
        = get_numpy_data(products, IMPORTANT_WORDS, 'sentiment')
    print(feature_matrix.shape)

    predict_probability(feature_matrix, np.zeros(feature_matrix.shape[1]))

    coefficients = run_lg(feature_matrix, label_array)
    prediction = predict(coefficients, feature_matrix)
    evaluate(coefficients, label_array, prediction)

if __name__ == '__main__':
    main()