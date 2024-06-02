import requests
from io import BytesIO
from typing import List


import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    # March 2023
    for year, months in [(2023, (3, 4))]:
        for month in range(*months):
            df = pd.read_parquet(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet')
            dfs.append(df)

        return pd.concat(dfs)
