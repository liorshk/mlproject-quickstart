import pandas as pd

from ml.data_source.base import DataSource
from ml.preprocessing.text_vectorizer import TextVectorizer
class Titanic(DataSource):

    def __init__(self, path) -> None:
        """        
        Parameters
        ----------            
        path : The path to the titantic csv
        """
        self._path = path

    def get_features(self)->pd.DataFrame:
        """
        Returns the features for the titanic data set

        Returns
        -------
        pd.DataFrame
            Dataframe with data
        """
        df = pd.read_csv(self._path)

        df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
        df['Is_Married'] = 0
        df['Is_Married'].loc[df['Title'] == 'Mrs'] = 1

        features = df[['Pclass', 'Age','Is_Married']]
        features.fillna(0, inplace=True)

        return features

    def get_label(self)->pd.Series:
        """
        Returns the label for the titanic data set

        Returns
        -------
        pd.Series
        """
        return pd.read_csv(self._path)['Survived']