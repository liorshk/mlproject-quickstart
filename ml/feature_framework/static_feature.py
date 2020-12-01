import os
from ml.feature_framework.feature import Feature
from util import hash_params, create_dirs
import pandas as pd
import inspect

class StaticFeature(Feature):
    def __init__(self, feature_function, **kwargs):
        super().__init__(feature_function, **kwargs)

    def _get_cached_path(self):        
        location = os.path.dirname(__file__)
        location = location[:location.rfind(os.path.sep)]
        feature_path = os.path.join(location,f'cache/static_features/{self.feature_name}')
        pickle_file_name = f'{hash_params(self._params_dict)}.pkl'
        full_feature_path = os.path.join(feature_path, pickle_file_name)
        
        return full_feature_path

    def run(self):
        
        full_feature_path  = self._get_cached_path()

        if os.path.exists(full_feature_path):
            df = pd.read_pickle(full_feature_path)
        else:
            df = self._feature_function(**self._params_dict)
            create_dirs(os.path.dirname(full_feature_path))
            
            df.to_pickle(full_feature_path)

        return df

def static_feature(feature_func):
    def wrapper(*args, **kwargs):
        func_args_names = [k for k in inspect.signature(feature_func).parameters]
        if feature_func.__defaults__ is not None:
            args_to_drop_count = len(feature_func.__defaults__)
            func_args_names = func_args_names[:-args_to_drop_count]

        additional_kwargs = dict(zip(func_args_names, args))
        kwargs.update(additional_kwargs)
        return StaticFeature(feature_func, **kwargs)
    return wrapper
