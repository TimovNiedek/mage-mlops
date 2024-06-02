import pandas as pd

def calculate_duration(df: pd.DataFrame) -> pd.DataFrame:
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    return df

def filter_duration_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    return df

def locations_to_str(df: pd.DataFrame) -> pd.DataFrame:
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df
