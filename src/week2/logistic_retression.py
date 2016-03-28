import numpy as np
from src.utils.utils import *


def predict(coefs, feature_matrix):
    product = feature_matrix.dot(coefs)
    return np.vectorize(lambda scalar: 1 if scalar > 0. else -1)(product)


def evaluate(coefs, labels, prediction):
    from sklearn.metrics import accuracy_score
    print('positive review count: {}'
          .format(np.count_nonzero(prediction + np.ones(prediction.shape))))
    print('accuracy: {}'.format(accuracy_score(labels, prediction)))


def predict_probability(feature_matrix, coefficients):
    score = feature_matrix.dot(coefficients)
    return vSigmoid(score)


def compute_log_likelihood(feature_matrix, sentiment, coefficients, l2_penalty):
    indicator = sentiment * 2 - 1
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores))) \
        - l2_penalty * np.sum(coefficients[1:] ** 2)
    return lp


def feature_derivative(errors, feature):
    derivative = feature.transpose().dot(errors)
    return derivative


def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter, l2_penalty=0, debug=True):
    coefficients = np.array(initial_coefficients)
    for itr in range(max_iter):
        predictions = predict_probability(feature_matrix, coefficients)
        indicator = (sentiment + 1) / 2

        errors = indicator - predictions
        derivative = feature_derivative(errors, feature_matrix) -2 * l2_penalty * coefficients
        coefficients += step_size * derivative

        if debug and (itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0)
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0):
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients, l2_penalty)
            print('iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp))
    return coefficients
