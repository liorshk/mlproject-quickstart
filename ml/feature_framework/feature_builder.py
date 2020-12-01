from datetime import datetime
import pandas as pd
from functools import reduce

from ml.feature_framework.static_feature import StaticFeature
from ml.feature_framework.time_based_feature import TimeBasedFeature

def build_features(feature_list: list, prediction_date: datetime, id_col: str, filter_func=None):

    if len(feature_list) == 0:
        raise Exception("There must be more than 1 features")

    dataframes = []
    for feature in feature_list:
        if isinstance(feature, StaticFeature):
            dataframes.append(feature.run())
        elif isinstance(feature, TimeBasedFeature):
            dataframes.append(feature.run(prediction_date))
        else:
            raise Exception(f"Features must be of types StaticFeature or TimeBasedFeature, given {type(feature)}")

    final_dataframe = reduce(lambda x,y: pd.merge(x,y,on=id_col), dataframes)

    if filter_func is not None:
        final_dataframe = final_dataframe[filter_func]

    final_dataframe.set_index(id_col, inplace=True)

    return final_dataframe