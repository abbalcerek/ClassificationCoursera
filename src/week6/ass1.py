from src.utils.utils import *
from src.utils.config import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, recall_score, precision_score, precision_recall_curve, confusion_matrix
import numpy as np


products = pd.read_csv(project_root('data/amazon_baby.csv'))

products['review_clean'] = products['review'].apply(remove_punctuation)
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1)

train_data, test_data = load_train_test_sets(products,
                                           "module-4-assignment-train-idx.json",
                                           "module-4-assignment-validation-idx.json")

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

train_matrix = vectorizer.fit_transform(train_data['review_clean'])
train_labels = train_data['sentiment']

test_matrix = vectorizer.transform(test_data['review_clean'])
test_labels = test_data['sentiment']

logistic_regression = LogisticRegression()
logistic_regression.fit(train_matrix, train_labels)

print("classes:", logistic_regression.classes_)


training_prediction = logistic_regression.predict(train_matrix)
prediction = logistic_regression.predict(test_matrix)


def asses_classifier(labels, prediction, classes, name=None):
    if name: print("\n=========={}=======".format(name))

    print("Accuracy on test data: {}".format(accuracy_score(labels, prediction)))
    print("Precision on test data: %s" % precision_score(labels, prediction))
    print("Recall on test data: %s" % recall_score(labels, prediction))
    print("F1 score on test data: %s" % f1_score(labels, prediction))

    cmat = confusion_matrix(labels,
                            prediction,
                            labels=classes)  # use the same order of class as the LR model.
    print(' target_label | predicted_label | count ')
    print('--------------+-----------------+-------')

    for i, target_label in enumerate(classes):
        for j, predicted_label in enumerate(classes):
            print('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i, j]))


asses_classifier(train_labels, training_prediction, logistic_regression.classes_, 'LR on training')
asses_classifier(test_labels, prediction, logistic_regression.classes_, 'LR on validation')
asses_classifier(test_labels, np.ones(len(test_labels)), logistic_regression.classes_, 'Majority on validation')

threshold = 0.9

threshold_prediction = [1 if i[1] > threshold else -1 for i in logistic_regression.predict_proba(test_matrix)]
asses_classifier(test_labels, threshold_prediction, logistic_regression.classes_, "threshold: {}".format(threshold))
