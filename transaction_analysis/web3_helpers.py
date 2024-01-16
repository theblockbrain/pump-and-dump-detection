import requests
import json


def _fetch_contract_list():
    URL = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"
    r = requests.get(url=URL)
    data = r.json()
    _write_dict_to_json_file("./coin_contracts.json", data)


def _read_contract_list():
    contract_path = "./coin_contracts.json"
    try:
        # Trying to open the file in read mode
        with open(contract_path, "r") as file:
            print("HERE")
            data = json.load(file)
    except FileNotFoundError:
        # If file is not found, create it
        with open(contract_path, "w") as file:
            _fetch_contract_list()
            data = json.load(file)
    return data


def _write_dict_to_json_file(outfile_name, data):
    json_data = json.dumps(data, indent=4)
    with open(outfile_name, "w") as outfile:
        outfile.write(json_data)


def get_contract_for_symbol(symbol: str):
    data = _read_contract_list()

    eth_address = ""
    for token in data:
        if token["symbol"] == symbol:
            if not "ethereum" in token["platforms"]:
                print(token["platforms"])
                print("No contract for ethereum specified")
                # raise ValueError("No contract for ethereum specified")
                # To-Do: Check if Alchemy allows algorand, ... data
            else:
                eth_address = token["platforms"]["ethereum"]
                # print(eth_address)
    # print(eth_address)
    return eth_address


def get_block_number_for_unix_date(unix_date: float, hex_return: bool):
    URL = f"https://coins.llama.fi/block/ethereum/{unix_date}"
    r = requests.get(url=URL)
    data = r.json()
    block_number = data["height"]

    if hex_return is True:
        block_number = hex(block_number)
    return block_number


def _make_alchemy_request(params: dict):
    alchemy_api_key = "4zWEsHMFRJ5CqkZLIp5m3322BRPbwB1q"
    my_url = f"https://eth-mainnet.g.alchemy.com/v2/{alchemy_api_key}"
    my_headers = {"Accept": "application/json", "Content-Type": "application/json"}

    r = requests.post(url=my_url, json=params, headers=my_headers)
    return r.json()


def get_file_from_alchemy_api(params: dict, outfile_name: str):
    alchemy_answer = _make_alchemy_request(params)
    print(alchemy_answer)

    if "error" in alchemy_answer:
        print("ERROR: ", alchemy_answer)
    else:
        transfers = alchemy_answer["result"]["transfers"]

        while {True}:
            if not "pageKey" in alchemy_answer["result"]:
                break

            print("Answer is paginated. Downloading more pages...")
            params["params"][0]["pageKey"] = alchemy_answer["result"]["pageKey"]
            alchemy_answer = _make_alchemy_request(params)
            transfers.append(alchemy_answer["result"]["transfers"])

        print("Number of returned transfers: ", len(transfers))
        _write_dict_to_json_file(outfile_name, transfers)
        print("Answer is complete")
