
from typing import Tuple

import scipy
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

def extract_features(df: pd.DataFrame) -> Tuple[scipy.sparse.csr_matrix, DictVectorizer]:
    categorical_cols = ['PULocationID', 'DOLocationID']

    train_dicts = df[categorical_cols].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    return X_train, dv