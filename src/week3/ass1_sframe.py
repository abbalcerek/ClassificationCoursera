# this gonna work only on python 2.7

import pandas as pd
import numpy as np
from src.utils.config import project_root
from src.utils.utils import load_train_test_sets
import sframe

from src.week3.OneHotTransformer import OneHotTransformer

pd.set_option('expand_frame_repr', False)

loans = sframe.SFrame(project_root('data/lending-club-data.gl/'))

# loans = pd.read_csv(project_root("data/lending-club-data.csv"))
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
loans = loans.remove_column('bad_loans')

safe_loans_count = len(loans[loans['safe_loans'] == 1])
print("safe loan count: ", float(safe_loans_count) / len(loans))

features = ['grade',  # grade of the loan
            'sub_grade',  # sub-grade of the loan
            'short_emp',  # one year or less of employment
            'emp_length_num',  # number of years of employment
            'home_ownership',  # home_ownership status: own, mortgage or rent
            'dti',  # debt to income ratio
            'purpose',  # the purpose of the loan
            'term',  # the term of the loan
            'last_delinq_none',  # has borrower had a delinquincy
            'last_major_derog_none',  # has borrower had 90 day or worse rating
            'revol_util',  # percent of available credit being used
            'total_rec_late_fee',  # total late fees received to day
            ]

target = 'safe_loans'  # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]

safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print("Number of safe loans  : %s" % len(safe_loans_raw))
print("Number of risky loans : %s" % len(risky_loans_raw))

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw) / float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)

percentage = len(risky_loans_raw) / float(len(safe_loans_raw))
risky_loans = risky_loans_raw

print("safe loans: ", len(safe_loans), "risky loans", len(risky_loans))

train_data, validation_data = loans_data.random_split(.8, seed=1)

print(train_data.shape, validation_data.shape)

train_data_d = train_data.to_dataframe()
validation_data_d = validation_data.to_dataframe()

print(train_data_d.shape)

print(train_data_d.dtypes)
print(type(train_data_d.dtypes[0].type))

categorical = [name for name, tp in zip(train_data_d.columns, train_data_d.dtypes)
               if not (issubclass(tp.type, np.integer) or issubclass(tp.type, np.float))]
print(categorical)

for name in categorical:
    print(name, set(loans_data[name]))

transformer = OneHotTransformer(categorical, train_data_d.columns)
transformer.fit(train_data_d)
train_data_transformed = transformer.transform_frame(train_data_d)
validation_data_transformed = transformer.transform_frame(validation_data_d)

print(train_data_transformed.shape, validation_data_transformed.shape)
print(list(train_data_transformed.columns))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix


# train data
training_labels = train_data_transformed[target]
del train_data_transformed[target]
# validation data
validation_labels = validation_data_transformed[target]

# sample validation
validation_safe2 = validation_data_transformed[validation_data_transformed[target] == 1][0:2]
validation_risky2 = validation_data_transformed[validation_data_transformed[target] == -1][0:2]
validation_set2 = validation_safe2.append(validation_risky2)
label_safe2 = validation_safe2[:2][target]
label_risky2 = validation_risky2[:2][target]
label2 = label_safe2.append(label_risky2)

validation_labels2 = validation_set2[target]
del validation_set2[target]

del validation_data_transformed[target]

print("======model depth={} sample data=========".format(6))
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(train_data_transformed, training_labels)
prediction = classifier.predict(validation_set2)
predict_proba = classifier.predict_proba(validation_set2)
print(prediction[:4])
print(predict_proba[:4])
export_graphviz(classifier)
print(accuracy_score(label2, prediction))


def run_classifier(depth):
    print("======model depth={}=========".format(depth))
    classifier = DecisionTreeClassifier(max_depth=depth)
    classifier.fit(train_data_transformed, training_labels)
    prediction = classifier.predict(validation_data_transformed)
    predict_proba = classifier.predict_proba(validation_data_transformed)
    print(prediction[:5])
    print(predict_proba[:5])
    export_graphviz(classifier)
    print(confusion_matrix(validation_labels, prediction))
    print(accuracy_score(validation_labels, prediction))


run_classifier(6)
run_classifier(2)
run_classifier(10)

print(1661 * 10000 + 1715 * 20000)
# small_model = DecisionTreeClassifier(max_depth=2)  # the best 5
# small_model.fit(train_data_transformed, training_labels)
# small_model_prediction = small_model.predict(validation_data_transformed)
# print(accuracy_score(validation_labels, small_model_prediction))
#
# big_model = DecisionTreeClassifier(max_depth=10)
# big_model.fit(train_data_transformed, training_labels)
# big_model_prediction = big_model.predict(validation_data_transformed)
# print(accuracy_score(validation_labels, big_model_prediction))
