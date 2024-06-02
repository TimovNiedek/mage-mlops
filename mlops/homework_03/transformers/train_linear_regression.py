from typing import Tuple

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from mlops.homework_03.utils.data_preparation.feature_extraction import extract_features

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def train(data, *args, **kwargs) -> Tuple[DictVectorizer, LinearRegression]:
    df = data['data_preparation'][0]
    
    X_train, dv = extract_features(df)
    y_train = df['duration'].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(lr.intercept_)

    return dv, lr