# enrich data with platform and contract
# WIP 05-06-2023

import pandas as pd
import json

pd_df = pd.read_csv("pump_telegram.csv")
pd_df = pd_df.head(2)

symbols = pd_df["symbol"].to_list()
print(symbols)

with open('coin_contracts.json', 'r') as openfile:
        coin_contracts = json.load(openfile)

platforms = list()
for symbol in symbols:
    if coin_contracts["symbol"] == symbol:
            

            if not "ethereum" in token["platforms"]:
                print(token['platforms'])
                raise ValueError("No contract for ethereum specified")
                # To-Do: Check if Alchemy allows algorand, ... data
            eth_address = token["platforms"]["ethereum"]

            