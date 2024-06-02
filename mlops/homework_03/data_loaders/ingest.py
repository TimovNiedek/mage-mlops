import requests
from io import BytesIO
from typing import List


import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    # Default : March 2023 (yellow)
    year = int(kwargs.get('year', '2023'))
    months = [
        int(month) 
        for month in 
        str(kwargs.get('months', '3')).split(',')
    ]
    color = kwargs.get('taxi_color', 'yellow')

    print(f'Fetching data for {color=}, {year=}, {months=}')

    for month in months:
        df = pd.read_parquet(f'https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet')
        dfs.append(df)

    return pd.concat(dfs)
