from abc import ABC, abstractmethod

class Feature(ABC):
    def __init__(self, feature_function, id_col, **kwargs):
        self._id_col = id_col
        self._params_dict = kwargs
        self._feature_function = feature_function

    @property
    def feature_name(self):
        return self._feature_function.__name__

    @property
    def id_col(self):
        return self._id_col

    @abstractmethod
    def run(self, *args):
        pass

