from src.utils.utils import *
from src.utils.config import *
import pandas as pd


IMPORTANT_WORDS = load_json_list(project_root("data/important_words.json"))


def explore_data(products):
    print('--------explore data--------')
    print('---10 first names---')
    print(products[:10]['name'])
    positive = [value for value in products['sentiment'] if value == 1]
    negative = [value for value in products['sentiment'] if value == -1]
    print('positive:', len(positive), ', negative:', len(negative))
    print('perfect count:', len([True for i in products['perfect'] if i > 0]))


def clean_data(products):
    products['review'] = products['review'].fillna('')
    revs = products['review']
    products['review_clean'] = products['review'].apply(remove_punctuation)

    for rev in revs:
        if not isinstance(rev, str):
            print(rev)


def add_word_counts(products):
    for word in IMPORTANT_WORDS:
        products[word] = products['review_clean'].apply(lambda s : s.split().count(word))


def set_up_products():
    from os.path import isfile
    from sklearn.externals import joblib
    path = project_root('data/serialized2/products')
    if isfile(path):
        return joblib.load(path)
    products = pd.read_csv(project_root('data/amazon_baby_subset.csv'))
    clean_data(products)
    add_word_counts(products)
    joblib.dump(products, path)
    return products


def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return feature_matrix, label_array


def main():
    products = set_up_products()
    explore_data(products)
    feature_matrix, label_array = get_numpy_data(products, IMPORTANT_WORDS, 'sentiment')
    print(feature_matrix.shape)

if __name__ == '__main__':
    main()