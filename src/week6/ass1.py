from src.utils.utils import *
from src.utils.config import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, recall_score, precision_score, precision_recall_curve, confusion_matrix


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

print "columns", vectorizer.get_feature_names()
print train_matrix.shape, test_matrix.shape

logistic_regression = LogisticRegression()
logistic_regression.fit(train_matrix, train_labels)

training_prediction = logistic_regression.predict(train_matrix)
prediction = logistic_regression.predict(test_matrix)


def asses_classifier(labels, prediction):
    print "accuracy", accuracy_score(labels, prediction)
    precision_recall_curve(labels, prediction)
    print confusion_matrix(labels, prediction, logistic_regression.classes_)
    print "Precision on test data: %s" % precision_score(labels, prediction)
    print "Recall on test data: %s" % recall_score(labels, prediction)

asses_classifier(train_labels, training_prediction)
asses_classifier(test_labels, prediction)

baseline = len(test_data[test_data['sentiment'] == 1])/len(test_data)
print "Baseline accuracy (majority class classifier): %s" % baseline