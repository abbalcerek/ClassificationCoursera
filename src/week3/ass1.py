import pandas as pd
import numpy as np
from src.utils.config import project_root
from src.utils.utils import load_train_test_sets

from src.week3.OneHotTransformer import OneHotTransformer

pd.set_option('expand_frame_repr', False)

loans = pd.read_csv(project_root("data/lending-club-data.csv"))
loans.insert(len(loans.columns), 'safe_loans', loans['bad_loans'].apply(lambda x: -1 if x == 1 else 1))
del loans['bad_loans']


value_counts = loans['safe_loans'].value_counts()
safe_loans_count = value_counts[1]
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

percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(frac=percentage, replace=False, random_state=1)

loans_data = risky_loans.append(safe_loans)
print(len(safe_loans), len(risky_loans))


print(loans_data.dtypes)
print(type(loans_data.dtypes[0].type))

categorical = [name for name, tp in zip(loans_data.columns, loans_data.dtypes)
               if not (issubclass(tp.type, np.integer) or issubclass(tp.type, np.float))]
print(categorical)

for name in categorical:
    print(name, set(loans_data[name]))


transformer = OneHotTransformer(categorical, loans_data.columns)
transformer.fit(loans_data)
fitted = transformer.transform_frame(loans_data)

train_set, validation_set \
    = load_train_test_sets(fitted, 'module-5-assignment-1-train-idx.json', 'module-5-assignment-1-validation-idx.json')

print(train_set.shape, validation_set.shape)
print(list(train_set.columns))

