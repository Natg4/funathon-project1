import duckdb
import os
import pandas as pd

RANDOM_STATE = 202605

# Create a non-persistent connection (the database exists only while the connection is alive and disappears when it is closed)
con = duckdb.connect(database=":memory:")


# We load all transactions made in France between 2010 and 2024
trans = con.sql(
    """
        SELECT * FROM read_parquet('https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/1_input/transactions_EN.parquet')
    """).to_df()



trans = trans[trans["prop_loc_dep"].isin(["75", "77", "78", "91", "92", "93", "94", "95"])]

trans["price_sqm"] = trans["price"] / trans["farea"]

trans = trans.dropna(subset = "price_sqm")

df = trans.drop(columns=[
    "price", "prop_loc_dep", "prop_loc_citycode", "dist_tosea"
])

# Filtering NA values
df = df.dropna()

df["prop_type"] = pd.Categorical(
    df["prop_type"].astype(str),
    categories=["1", "2"],
    ordered=False
).rename_categories({"1": "House", "2": "Flat"})

df['prop_year_harm_10'] = (df['prop_year_harm'] // 10)*10
df['prop_year_harm_10'] = df['prop_year_harm_10'].where(df['prop_year_harm_10'] >= 1850, 1840)

# Dropping old column
df = df.drop(columns=["prop_year_harm"])




