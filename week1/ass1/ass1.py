import pandas as pd
import re

pd.set_option('expand_frame_repr', False)
products = pd.read_csv('data/amazon_baby.csv')


def remove_punctuation(text):
    from string import punctuation
    if pd.isnull(text): return text
    return re.sub('[{}]'.format(punctuation), '', text)

products['review_clean'] = products['review'].apply(remove_punctuation)

print(products)
