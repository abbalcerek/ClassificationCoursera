from src.utils.config import *


def hello():
    print("hello world")


def split_dataframe(df, fraction=.8):
    pass


def load_json_list(file_name):
    with open(file_name) as file:
        import json
        return json.load(file)


def remove_punctuation(text):
    from string import punctuation
    import pandas as pd
    import re
    if pd.isnull(text):
        return ''
    return re.sub('[{}]'.format(punctuation), '', text)



def sigmoid(x):
  import math
  return 1 / (1 + math.exp(-x))


def vSigmoid(array):
    import numpy as np
    return np.vectorize(sigmoid)(array)


def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return feature_matrix, label_array


def load_train_test_sets(products, train_path, test_path):
    from os.path import join
    train_data_indexes = load_json_list(project_root(join("data", train_path)))
    train_data_filtered_index = sorted(list(set(products.index).intersection(set(train_data_indexes))))
    print(train_data_filtered_index)
    print(sorted(products.index))
    train_data = products.iloc[train_data_filtered_index]
    test_data_indexes = load_json_list(project_root(join("data", test_path)))
    test_data_filtered_indexes = set(products.index).intersection(test_data_indexes)
    test_data = products.iloc[products.index]
    return train_data, test_data

