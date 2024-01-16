# autoencoder_pd_detection
This repository holds the code developed for my master thesis "Unmasking Cryptocurrency Pump and Dump Schemes:
An Autoencoder Approach for Detection"

```python 
autoencoder_pd_detection
├── README.md
├── requirements.txt
├── data
│   ├── combine_csvs.ipynb #combines La Morgia Data & NatSD into one csv
│   ├── combined #combined csvs           
│   ├── la_morgia_data #La Morgia Data
│   └── natural_sd #NatSD Data
├── models
│   ├── naive_bayes.ipynb
│   ├── autoencoder
│   │   ├── ae_evaluate_model.ipynb #calculates plots for the best ae model
│   │   ├── ae_models.py #holds different architectures
│   │   ├── ae_sweep.py #conducts the random search to find the best ae model
│   │   ├── best_model #weights and config for best ae model found by us
│   │   └── random_search_conf_ae.yml #config for ae_sweep.py
│   └── vae
│       ├── best_model #weights and config for best vae model found by us
│       ├── keras_vae_sweep.py #conducts the random search to find the best vae model
│       ├── random_search_conf_vae.yml #config for keras_vae_sweep.py
│       └── vae_evaluate_model.ipynb #calculates plots for the best vae model
├── natsd_dataset             
│   ├── analyze_data.ipynb #calculates plots and statistics for the finished NatSD data
│   ├── download_price_data.ipynb #downloads raw price data from binance
│   ├── build_dataset.ipynb #builds NatSD dataset from raw price data
│   ├── la_morgia_features.py #used in build_dataset to calculate the features
├── plots                     
│   ├── mean_spike_time.ipynb #plots mean spike times figure
│   └── teaserfigure.ipynb #plots the teaserfigure
└── transaction_analysis
    ├── transfers_overall.ipynb #conducts the qualitative transaction analysis
    └── web3_helpers.py #helper functions for transfers_overall.ipynb
```


## Installation
Install the requirements for this repo:
```
pip3 install -r requirements.txt
```

## Run Autoencoder / Variational Autoencoder
Our models are stored in the `models` folder.

To run one of the models go into the specific folder.
With the `<ae/vae>_evalute_model.ipynb` notebook the model can be tested with a specific configuration that can be given to him using the configuration in the top of the file.
This notebook also creates the reconstruction errors and the latent space plots found in our paper.

The random search for the hyperparameter optimzation was conducted using the [weights and biases](https://wandb.ai/site) library. The code for this can be found in the `<ae/vae>_sweep.py` files.
To run the random search weights and biases needs to be setup as described in their [quickstart](https://docs.wandb.ai/quickstart).
Inside the `random_search_conf_<a/vae>.yml` the search space for the random search can be defined.

The best configurations found by us are given in the `best_model/` folder.


## Build NatSD Dataset
The computed features for the NatSD Dataset can be found in the `data` folder.

To build the NatSD Dataset from scratch go into the `data` folder.

With `download_price_data.ipynb` the raw price data from Binance API can be downloaded. The desired coin-pairings and time-range can be specified there.

After this `build_dataset.ipynb` can be used to find the maximum drawdowns / spikes and compute the features using the code provided by La Morgia et al. (see `la_morgia_features.py`).

With `analyze_data.ipynb` the raw price data can be checked for gaps the Binance API has in the specified time-range. Also some statistics and plots can be computed.


## Baselines

[Random Forest model](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset) published by La Morgia et al. 

[Anomaly Transformer and C-LSTM model](https://github.com/Derposoft/crypto_pump_and_dump_with_deep_learning) published by Chadalapaka et al.
To run the Anomaly Transformer we used this command
```
python -m train --model AnomalyTransformer --feature_size 12 --n_layers 4 --n_epochs 200 --lambda 0.0001 --lr 1e-3 --lr_decay_step 0 --batch_size 32 --undersample_ratio 0.05 --segment_length 15 --prthreshold 0.48 --kfolds 5 --dataset ./data features_15S.csv.gz --final_run True
```
And for the C-LSTM this
```
python -m train --model CLSTM --embedding_size 350 --n_layers 1 --n_epochs 200 --kernel_size 3 --dropout 0.0 --cell_norm False --out_norm False --lr 1e-3 --lr_decay_step 0 --batch_size 600 --undersample_ratio 0.05 --segment_length 15 --prthreshold 0.7 --kfolds 5 --dataset ./data/features_15S.csv.gz --final_run True
```
based on their configurations published [here](https://github.com/Derposoft/crypto_pump_and_dump_with_deep_learning/tree/main/models).

