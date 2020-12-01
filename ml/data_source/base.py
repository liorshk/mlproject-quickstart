from abc import ABC, abstractmethod
import pandas as pd

class DataSource(ABC):
    
    @abstractmethod
    def get_features(self) -> pd.DataFrame:
        """
        Gets the features for the dataset
        """
        pass
         
    @abstractmethod
    def get_label(self) -> pd.Series:
        """
        Gets the label for the dataset
        """
        pass
         
