import pandas as pd
from datetime import datetime
import time
import json
from web3_lib.web3 import get_contract_for_symbol
from web3_lib.web3 import get_block_number_for_unix_date
from web3_lib.web3 import get_file_from_alchemy_api

"""
Worked until

Index 885 of 1110
TNB 2020-03-21 16:00
Answer is complete
Number of returned transfers:  22
"""

pd_df = pd.read_csv("pump_telegram.csv")
# pd_df = pd_df.head(10)
rows = len(pd_df. index)

for index, row in pd_df.iterrows():
    print(f"Index {index} of {rows}")
    print(row['symbol'], row['date'], row["hour"])
    symbol = row['symbol'].lower()
    event_timestamp = row['date'] + ' ' + row ['hour']

    eth_address = get_contract_for_symbol(symbol)

    if eth_address != "":
        start_date = datetime.strptime(event_timestamp, "%Y-%m-%d %H:%M")
        start_date = start_date.replace(hour=0, minute=0)
        unix_start_date = time.mktime(start_date.timetuple())


        end_date = datetime.strptime(event_timestamp, "%Y-%m-%d %H:%M")
        end_date = start_date.replace(hour=23, minute=59)
        unix_end_date = time.mktime(end_date.timetuple())

        hex_block_number_start = get_block_number_for_unix_date(unix_start_date, True)
        hex_block_number_end = get_block_number_for_unix_date(unix_end_date, True)

        params = {'jsonrpc': '2.0', 'method': 'alchemy_getAssetTransfers', 'params': [
            {
            "fromBlock": hex_block_number_start,
            "toBlock": hex_block_number_end,
            "contractAddresses": [eth_address],
            "category": ["erc20"],
            "withMetadata": True,
            "excludeZeroValue": True,
            # "maxCount": "0x3e8" # maximum transactions possible
            }
        ]}

        answer_file = f"./alchemy_data/alchemy_answer_{symbol}.json"
        get_file_from_alchemy_api(params, answer_file)

        with open(answer_file, 'r') as openfile:
            json_data = json.load(openfile)

        print("Number of returned transfers: ", len(json_data))

    else:
        print("Cannot download this data")

    print(" ")

