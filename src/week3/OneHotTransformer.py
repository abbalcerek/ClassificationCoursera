import pandas as pd


class OneHotEncoder():
    def __init__(self, categorical, columns):
        self.__columns = columns
        self.__categorical = categorical
        self.__values = {}

    def fit(self, dataframe):
        for feature in self.__categorical:
            self.__values[feature] = sorted(list(set(dataframe[feature])))

    def __create_dataframe_for_feature(self, feature, dataframe):
        result = dataframe[[feature]]
        for value in self.__values[feature]:
            result.insert(len(result.columns),
                          '_'.join([feature, value]),
                          result.ix[:, 0].apply(lambda row_value: 1 if value == row_value else 0)
                          )
        del result[feature]
        return result

    def transform_frame(self, dataframe):
        frames = [self.__create_dataframe_for_feature(feature, dataframe) for feature in self.__categorical]
        diff = set(dataframe.columns.values).difference(self.__categorical)
        return pd.concat([dataframe[list(diff)]] + frames, axis=range(4))

    def transform_one(self, row):
        frame = pd.DataFrame([row], columns=self.__columns, index=[0])
        print('frame\n', frame)
        return self.transform_frame(frame)

# df = pd.DataFrame({'x' : ["a", "b", "a", "c"], 'z' : ["a", "c", "c", "c"], 'y' : [4, 5, 6, 7]})
# print(df)
#
#
# enc = OneHotEncoder(['x', 'z'], ['x', 'y', 'z'])
# enc.fit(df)
#
# print(enc.transform_frame(df))
# print(enc.transform_frame(df))
#
# print(enc.transform_one(['a', 4, 'a']))
#
#
