import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    f1_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
)
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import yaml
import pprint
from keras.callbacks import LambdaCallback, Callback
from tensorflow.keras.optimizers import Adam, SGD
from ae_models import Autoencoder_8_4, Autoencoder_8_6_4, Autoencoder_8_7_5


def create_datasets():
    time_freq = "15S"
    computed_data = pd.read_csv(
        "../../data/la_morgia_data/features_" + time_freq + ".csv.gz",
        parse_dates=["date"],
    )

    normal_datapoints = computed_data[computed_data["gt"] == 0]
    anomaly_datapoints = computed_data[computed_data["gt"] == 1]

    # randomness
    train_set = normal_datapoints.sample(frac=0.95, random_state=1)

    test_set_normal = normal_datapoints.drop(train_set.index)
    test_sets = [test_set_normal, anomaly_datapoints]
    test_set = pd.concat(test_sets)

    features = [
        "std_rush_order",
        "avg_rush_order",
        "std_trades",
        "std_volume",
        "avg_volume",
        "std_price",
        "avg_price",
        "avg_price_max",
        # "hour_sin",
        # "hour_cos",
        # "minute_sin",
        # "minute_cos",
    ]

    x_train = train_set[features]
    x_test = test_set[features]
    y_train = train_set["gt"]
    y_test = test_set["gt"]

    return x_train, y_train, x_test, y_test


def set_optimizer(optimizer, learning_rate):
    if optimizer == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = SGD(learning_rate=learning_rate)

    return optimizer


def find_threshold(model, x_train):
    reconstructions = model.predict(x_train)
    reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train)

    threshold = np.mean(reconstruction_errors.numpy()) + np.std(
        reconstruction_errors.numpy()
    )
    return threshold


def get_predictions(model, x_test, threshold):
    reconstructions = model.predict(x_test)
    errors = tf.keras.losses.msle(reconstructions, x_test)

    # 1 = anomaly, 0 = normal
    anomaly_mask = pd.Series(errors) > threshold
    preds = anomaly_mask.map(lambda x: 1.0 if x == True else 0.0)
    return preds, errors


def log_metrics(predictions, threshold):
    overall_score = (
        accuracy_score(y_test, predictions)
        + precision_score(y_test, predictions)
        + recall_score(y_test, predictions)
        + f1_score(y_test, predictions, average="macro")
        + f1_score(y_test, predictions, average="micro")
    )

    wandb.log(
        {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1_Macro": f1_score(y_test, predictions, average="macro"),
            "F1_Micro": f1_score(y_test, predictions, average="micro"),
            "Overall": overall_score,
            "Confusion Matrix": confusion_matrix(y_test, predictions),
            "threshold": threshold,
        }
    )


def plot_reconstruction_errors(reconstruction_errors, run_name):
    errors = tf.Variable(reconstruction_errors).numpy().tolist()
    labels = list(y_test)
    offsets = np.random.uniform(-0.1, 0.1, size=len(errors))

    errors_df = pd.DataFrame(
        {f"reconstruction_error": errors, "x_offset": offsets, "label": labels}
    )
    errors_df = errors_df.sort_values(by="label", ascending=False)
    errors_table = wandb.Table(dataframe=errors_df)

    fields = {"x": "x_offset", "y": "reconstruction_error", "groupKeys": "label"}
    plot_name = f"Reconstruction Error {run_name}"
    wandb.log(
        {
            plot_name: wandb.plot_table(
                vega_spec_name="denis-reibel/reconstruction_errors",
                data_table=errors_table,
                fields=fields,
                string_fields={"title": f"Reconstruction Errors {run_name}"},
            )
        }
    )


class LogMetricsCallback(Callback):
    def __init__(self, autoencoder, run_name, log_frequency):
        super().__init__()
        self.autoencoder = autoencoder
        self.log_frequency = log_frequency
        self.run_name = run_name

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_frequency == 0:
            evaluate_model(self.autoencoder, self.run_name)


log_loss_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: wandb.log({"epoch": epoch, "loss": logs["loss"]})
)


def evaluate_model(autoencoder, run_name, plot=False):
    threshold = find_threshold(autoencoder, x_train)
    predictions, reconstruction_errors = get_predictions(
        autoencoder, x_test, 2 * threshold
    )
    log_metrics(predictions, threshold)

    if plot == True:
        plot_reconstruction_errors(reconstruction_errors, run_name)


def set_model_architechture(architecture, input_units, latent_dimension, activation):
    if architecture == "8_6_4":
        return Autoencoder_8_6_4(input_units, latent_dimension, activation)
    if architecture == "8_7_5":
        return Autoencoder_8_7_5(input_units, latent_dimension, activation)
    if architecture == "8_4":
        return Autoencoder_8_4(input_units, latent_dimension, activation)


def start_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        run_name = wandb.run.name

        input_units = x_train.shape[1]
        autoencoder = set_model_architechture(
            config.architecture, input_units, config.latent_dimension, config.activation
        )
        optimizer = set_optimizer(config.optimizer, config.learning_rate)
        autoencoder.compile(loss=config.loss, optimizer=optimizer)
        autoencoder.fit(
            x_train,
            x_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=(x_test, x_test),
            callbacks=[
                WandbMetricsLogger(log_freq=5),
                WandbModelCheckpoint("models"),
                log_loss_callback,
                LogMetricsCallback(autoencoder, run_name, 5),
            ],
        )
        evaluate_model(autoencoder, run_name, plot=True)
        autoencoder.save_weights(f"./weights/ae-sweep-10-11/{run_name}_weights.h5")

    wandb.finish()


if __name__ == "__main__":
    with open("./random_search_conf_ae.yml", "r") as f:
        sweep_config = yaml.safe_load(f)
    # pprint.pprint(sweep_config)
    x_train, y_train, x_test, y_test = create_datasets()
    sweep_id = wandb.sweep(sweep_config, project="ae-sweep-10-11")
    wandb.agent(sweep_id, start_sweep, count=200)
