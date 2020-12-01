import os
from ml.feature_framework.feature import Feature
from util import hash_params, create_dirs
import pandas as pd
from datetime import datetime
import inspect


class TimeBasedFeature(Feature):
    def __init__(self, feature_function, date_col='day_date', **kwargs):
        super().__init__(feature_function, **kwargs)
        self._date_col = date_col

    @property
    def date_col(self):
        return self._date_col

    def _get_cached_path(self):
        location = os.path.dirname(__file__)
        location = location[:location.rfind(os.path.sep)]
        return os.path.join(location,f'cache/time_based_features/{self.feature_name}/{hash_params(self._params_dict)}')

    def run(self, prediction_date: datetime):
      
        full_feature_path = self._get_cached_path()

        if os.path.exists(full_feature_path):
            df = pd.read_pickle(full_feature_path)
        else:
            df = self._feature_function(prediction_date, **self._params_dict)            
            create_dirs(full_feature_path)
            
            df.to_pickle(full_feature_path)
            
        return df


def time_based_feature(feature_func):
    def wrapper(*args, **kwargs):
        func_args_names = [k for k in inspect.signature(feature_func).parameters][2:]
        if feature_func.__defaults__ is not None:
            args_to_drop_count = len(feature_func.__defaults__)
            func_args_names = func_args_names[:-args_to_drop_count]

        additional_kwargs = dict(zip(func_args_names, args))
        kwargs.update(additional_kwargs)
        return TimeBasedFeature(feature_func, **kwargs)
    return wrapper