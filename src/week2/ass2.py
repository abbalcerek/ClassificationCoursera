from src.utils.utils import *
from src.utils.config import *
import pandas as pd
import numpy as np
from src.week2.logistic_retression import logistic_regression, evaluate, predict, compute_log_likelihood
from src.week2.ass1 import IMPORTANT_WORDS, set_up_products, print_important_words
from src.week2.plot_features import make_coefficient_plot


def run_model(train_matrix, labels, test_feature_matrix, test_label_array, l2_penalty):
    initial_coefficients = np.zeros(train_matrix.shape[1])
    step_size = 5e-6
    max_iter = 501
    coefs = logistic_regression(train_matrix, labels, initial_coefficients, step_size, max_iter, l2_penalty, False)
    train_prediction = predict(coefs, train_matrix)
    test_prediction = predict(coefs, test_feature_matrix)
    print("------penalty={}---------".format(l2_penalty))
    print("---train")
    evaluate(coefs, labels, train_prediction)
    print('llikelihood: {}'.format(compute_log_likelihood(train_matrix, labels, coefs, l2_penalty)))
    print("---test")
    evaluate(coefs, test_label_array, test_prediction)
    print('llikelihood: {}'.format(compute_log_likelihood(test_feature_matrix, test_label_array, coefs, l2_penalty)))
    print_important_words(coefs, 5)
    return coefs


def main():
    products = set_up_products()
    train_set, test_set = load_train_test_sets(products,
                                               "module-4-assignment-train-idx.json",
                                               "module-4-assignment-validation-idx.json")
    train_feature_matrix, train_label_array \
        = get_numpy_data(train_set, IMPORTANT_WORDS, 'sentiment')
    test_feature_matrix, test_label_array \
        = get_numpy_data(test_set, IMPORTANT_WORDS, 'sentiment')

    penalties = [0, 4, 10, 1e2, 1e3, 1e5]
    coefs = []
    for penalty in penalties:
        coefs.append(
            run_model(train_feature_matrix, train_label_array, test_feature_matrix, test_label_array, penalty))

    print("======word coef diagram=========")
    sorted_pairs = sorted(zip(coefs[0][1:], IMPORTANT_WORDS))
    sorted_words = [word for (value, word) in sorted_pairs]
    top_positive_wrods, top_negative_words = sorted_words[:5], sorted_words[-5:]

    dicts = [{word: value for (value, word) in zip(coef[1:], IMPORTANT_WORDS)
          if word in top_negative_words or word in top_positive_wrods} for coef in coefs]
    frames = [pd.DataFrame(d, index=[penalty]) for (penalty, d) in zip(penalties, dicts)]
    table = pd.concat(frames)

    make_coefficient_plot(table, top_positive_wrods, top_negative_words, l2_penalty_list=[0, 4, 10, 1e2, 1e3, 1e5])


if __name__ == '__main__':
    import sys
    import os
    pd.set_option('expand_frame_repr', False)
    path = project_root("data/logs")
    if not os.path.exists(path):
        os.makedirs(path)
    sys.stdout = open(os.path.join(path, "week2ass2.txt"), 'w')
    main()
