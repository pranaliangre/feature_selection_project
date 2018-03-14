
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(dataframe):
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
    model = RandomForestClassifier()

    rfe = RFE(model, round(len(X.columns) / 2, 0)).fit(X, y)
    top_features = []

    for i in range(len(rfe.ranking_)):
        if rfe.ranking_[i] == 1:
            top_features.append(X.columns[i])

    return top_features

