from src.utils.utils import *
from src.utils.config import *
import pandas as pd
import numpy as np
from src.week2.logistic_retression import logistic_regression, evaluate, predict


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
            .apply(lambda s: s.split().count(word))


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


def run_lg(feature_matrix, labels):
    initial_coefs = np.zeros(feature_matrix.shape[1])
    return logistic_regression(feature_matrix, labels, initial_coefs, 1e-7, 301)


def important_words(coefs, words, best=True, number=10):
    print('------------important words ---------------')
    sorted_pairs = sorted(zip(coefs[1:], words), reverse=best)
    for (coef, word) in sorted_pairs[0: number]:
        print('weight of word {}: {}'.format(word, coef))
    return sorted_pairs[0: number]


def print_important_words(coefs, number=10):
    positive = important_words(coefs, IMPORTANT_WORDS, True, number)
    negative = important_words(coefs, IMPORTANT_WORDS, False, number)
    return positive, negative


def main():
    products = set_up_products()
    explore_data(products)
    feature_matrix, label_array \
        = get_numpy_data(products, IMPORTANT_WORDS, 'sentiment')
    print(feature_matrix.shape)
    coefficients = run_lg(feature_matrix, label_array)
    prediction = predict(coefficients, feature_matrix)
    evaluate(coefficients, label_array, prediction)
    print_important_words(coefficients)


if __name__ == '__main__':
    main()