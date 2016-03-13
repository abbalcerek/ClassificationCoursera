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
