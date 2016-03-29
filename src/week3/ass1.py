import pandas as pd
import numpy as np
from src.utils.config import project_root
from src.utils.utils import load_train_test_sets
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

pd.set_option('expand_frame_repr', False)

loans = pd.read_csv(project_root("data/lending-club-data.csv"))

loans.insert(len(loans.columns), 'safe_loans', loans['bad_loans'].apply(lambda x: -1 if x == 1 else 1))
del loans['bad_loans']

value_counts = loans['safe_loans'].value_counts()
safe_loans = value_counts[1]
print("safe loan count: ", float(safe_loans) / len(loans))

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

train_set, validation_set \
    = load_train_test_sets(loans, 'module-5-assignment-1-train-idx.json', 'module-5-assignment-1-validation-idx.json')

loans = loans[features]
# print(loans.describe)
print(loans.dtypes)

categorical = [name for name, tp in zip(loans.columns, loans.dtypes) if tp not in (np.int, np.float)]
print(categorical)

for name in categorical:
    print(name, set(loans[name]))

categorical_indexes = [i for i in range(len(loans.columns)) if loans.columns[i] in categorical]
print(categorical_indexes)

for feature in categorical:
    loans_data_one_hot_encoded = loans[feature].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)

    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans.remove_column(feature)
    loans.add_columns(loans_data_unpacked)
