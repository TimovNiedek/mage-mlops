from typing import Tuple

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def train(df: pd.DataFrame, **kwargs) -> Tuple[DictVectorizer, LinearRegression]:
    categorical = ['PULocationID', 'DOLocationID']

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    y_train = df['duration'].values
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(lr.intercept_)

    return dv, lr