from typing import Tuple

from pathlib import Path
import tempfile
import pickle
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

mlflow.set_tracking_uri("http://mlflow:5000")

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def log_model(data: Tuple[DictVectorizer, LinearRegression], *args, **kwargs) -> None:
    dv, lr = data

    with tempfile.TemporaryDirectory() as tmp_dir, mlflow.start_run():
        tmp_path = Path(tmp_dir) / 'dict_vectorizer.pkl'
        with open(tmp_path, 'wb') as f:
            pickle.dump(dv, f)
    
    
        mlflow.sklearn.log_model(lr, 'linear_regression')
        mlflow.log_artifact(tmp_path, 'dict_vectorizer')
