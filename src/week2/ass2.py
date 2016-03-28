from src.utils.utils import *
from src.utils.config import *
import pandas as pd
import numpy as np
from src.week2.logistic_retression import logistic_regression, evaluate, predict
from src.week2.ass1 import IMPORTANT_WORDS, set_up_products


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
    print("---test")
    evaluate(coefs, test_label_array, test_prediction)


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
    for penalty in penalties:
        run_model(train_feature_matrix, train_label_array, test_feature_matrix, test_label_array, penalty)


if __name__ == '__main__':
    main()