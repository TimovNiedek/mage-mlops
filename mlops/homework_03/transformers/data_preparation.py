import pandas as pd
from mlops.homework_03.utils.data_preparation.preprocessing import (
    calculate_duration,
    filter_duration_outliers,
    locations_to_str,
)

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform_df(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df = calculate_duration(df)
    df = filter_duration_outliers(df)
    df = locations_to_str(df)
    return df
