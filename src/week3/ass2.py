import sframe
from src.utils.config import project_root
from src.week3.OneHotTransformer import OneHotTransformer

loans = sframe.SFrame(project_root('data/lending-club-data.gl/'))

loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
loans = loans.remove_column('bad_loans')

features = [
    'grade',  # grade of the loan
    'term',  # the term of the loan
    'home_ownership',  # home_ownership status: own, mortgage or rent
    'emp_length',  # number of years of employment
]
target = 'safe_loans'

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw) / float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed=1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print("Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data)))
print("Percentage of risky loans                :", len(risky_loans) / float(len(loans_data)))
print("Total number of loans in our new dataset :", len(loans_data))

encoder = OneHotTransformer(features, loans.column_names)
