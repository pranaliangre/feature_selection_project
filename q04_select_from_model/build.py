
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here

def select_from_model(dataframe):
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
    np.random.seed(9)
    model = RandomForestClassifier()

    sfm = SelectFromModel(model)
    sfm = sfm.fit(X, y)

    feature_idx = sfm.get_support()
    feature_name = X.columns[feature_idx]

    return list(feature_name)

