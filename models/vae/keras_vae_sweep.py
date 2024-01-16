import pandas as pd
import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    f1_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
)
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import LambdaCallback, Callback
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pprint


def create_datasets():
    time_freq = "15S"
    computed_data = pd.read_csv(
        "../_labeled_features/features_" + time_freq + ".csv.gz", parse_dates=["date"]
    )

    normal_datapoints = computed_data[computed_data["gt"] == 0]
    anomaly_datapoints = computed_data[computed_data["gt"] == 1]

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


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        # ToDo: Hyper tune sampling ?
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(activation, dimensions, latent_dim=2):
    inputs = layers.Input(shape=dimensions[0], name="encoder_input")
    x = layers.Dense(dimensions[1], activation=activation)(inputs)

    if len(dimensions) == 3:
        x = layers.Dense(dimensions[2], activation=activation)(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder


def build_decoder(activation, dimensions, latent_dim=2):
    latent_inputs = layers.Input(shape=(latent_dim,), name="z_sampling")

    if len(dimensions) == 3:
        x = layers.Dense(dimensions[2], activation=activation)(latent_inputs)
        x = layers.Dense(dimensions[1], activation=activation)(x)
    else:
        x = layers.Dense(dimensions[1], activation=activation)(latent_inputs)

    outputs = layers.Dense(dimensions[0])(x)

    decoder = keras.Model(latent_inputs, outputs, name="decoder")
    decoder.summary()
    return decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, kl_weight, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            # ToDo: reconstruction_loss / kl_loss gewichten ?
            total_loss = reconstruction_loss + self.kl_weight * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


log_loss_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: wandb.log({"epoch": epoch, "loss": logs["loss"]})
)


class LogMetricsCallback(Callback):
    def __init__(self, vae, log_frequency):
        super().__init__()
        self.vae = vae
        self.log_frequency = log_frequency

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_frequency == 0:
            evaluate_model(self.vae)


def evaluate_model(vae):
    error_thresh = get_error_thresh(vae)
    mae_vector, anomalies = get_anomalies(vae, error_thresh)
    log_metrics(anomalies, error_thresh)


def evaluate_model_final(vae, run_name, latent_dimension):
    error_thresh = get_error_thresh(vae)
    mae_vector, anomalies = get_anomalies(vae, error_thresh)
    log_metrics(anomalies, error_thresh)
    log_plots(vae, run_name, mae_vector, anomalies, latent_dimension)


def train_model(config):
    print("Config Dimensions: ", config.architecture)
    print(type(config.architecture))
    encoder = build_encoder(
        config.activation, config.architecture, latent_dim=config.latent_dimension
    )
    decoder = build_decoder(
        config.activation, config.architecture, latent_dim=config.latent_dimension
    )

    vae = VAE(encoder, decoder, config.kl_weighting)
    optimizer = set_optimizer(config.optimizer, config.learning_rate)
    vae.compile(optimizer=optimizer)
    vae.fit(
        x_train,
        x_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[
            log_loss_callback,
            WandbMetricsLogger(log_freq=5),
            LogMetricsCallback(vae, 5),
        ],
    )
    return vae


def get_error_thresh(vae):
    # Encode the input data to obtain the latent representations
    z_mean, z_log_var, z = vae.encoder.predict(x_train)
    x_train_reconstructions = vae.decoder.predict(z)

    mae_vector = np.mean(abs(x_train_reconstructions - x_train), axis=1)

    print(
        f"Avg error {np.mean(mae_vector)}\n median error {np.median(mae_vector)}\n 99Q: {np.quantile(mae_vector, 0.99)}"
    )
    print(f"setting threshold on { np.quantile(mae_vector, 0.9965)} ")

    error_thresh = np.quantile(mae_vector, 0.9965)
    return error_thresh


def get_anomalies(vae, error_thresh):
    # Encode the input data to obtain the latent representations
    z_mean, z_log_var, z = vae.encoder.predict(x_test)
    x_test_reconstructions = vae.decoder.predict(z)

    mae_vector = np.mean(abs(x_test_reconstructions - x_test), axis=1)
    anomalies = mae_vector > error_thresh
    return mae_vector, anomalies


def log_metrics(anomalies, error_thresh):
    predictions = anomalies
    threshold = error_thresh
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
            "Overall_Score": overall_score,
            "Confusion Matrix": confusion_matrix(y_test, predictions),
            "threshold": threshold,
        }
    )


def plot_latent_space(vae, run_name):
    z_mean, z_log_var, z = vae.encoder.predict(x_test)
    labels = y_test

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=z[:, 0], y=z[:, 1], s=20, hue=labels, palette="viridis")
    plt.title("Visualization of Latent Space")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True)

    image_path = f"./tmp_img/latent_space_{run_name}.png"
    plt.savefig(image_path)
    wandb.log({f"latent_space_{run_name}": wandb.Image(image_path)})
    plt.close()


def plot_vae_latent_space_reconstruction_errors(X_transform, mae_vector, run_name):
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=X_transform[:, 0], y=X_transform[:, 1], s=20, hue=mae_vector)
    plt.grid()

    image_path = f"./tmp_img/vae_latent_space_reconstruction_errors_{run_name}.png"
    plt.savefig(image_path)
    wandb.log(
        {f"vae_latent_space_reconstruction_errors_{run_name}": wandb.Image(image_path)}
    )
    plt.close()


def plot_vae_latent_space_anomalies(X_transform, anomalies, run_name):
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=X_transform[:, 0], y=X_transform[:, 1], s=20, hue=anomalies)
    plt.grid()

    image_path = f"./tmp_img/vae_latent_space_anomalies_{run_name}.png"
    plt.savefig(image_path)
    wandb.log({f"vae_latent_space_anomalies_{run_name}": wandb.Image(image_path)})
    plt.close()


def plot_vae_latent_space_anomalies_ground_truth(X_transform, run_name):
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=X_transform[:, 0], y=X_transform[:, 1], s=10, hue=y_test)
    plt.grid()

    image_path = f"./tmp_img/vae_latent_space_anomalies_ground_truth_{run_name}.png"
    plt.savefig(image_path)
    wandb.log(
        {f"vae_latent_space_anomalies_ground_truth_{run_name}": wandb.Image(image_path)}
    )
    plt.close()


def log_plots(vae, run_name, mae_vector, anomalies, latent_dimension):
    X_transform = vae.encoder.predict(x_test)[2]  # [0] z_mean, [1] z_log_var, [2] z
    if latent_dimension > 2:
        pca = PCA(n_components=2)
        X_transform = pca.fit_transform(X_transform)

    if latent_dimension != 1:
        plot_latent_space(vae, run_name)
        plot_vae_latent_space_reconstruction_errors(X_transform, mae_vector, run_name)
        plot_vae_latent_space_anomalies(X_transform, anomalies, run_name)
        plot_vae_latent_space_anomalies_ground_truth(X_transform, run_name)


def start_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        run_name = wandb.run.name

        vae = train_model(config)
        evaluate_model_final(vae, run_name, config.latent_dimension)
        vae.save_weights(f"./weights_vae/{run_name}_weights.h5")
    wandb.finish()


if __name__ == "__main__":
    with open("./random_search_conf_vae.yml", "r") as f:
        sweep_config = yaml.safe_load(f)
    pprint.pprint(sweep_config)
    x_train, y_train, x_test, y_test = create_datasets()
    sweep_id = wandb.sweep(sweep_config, project="keras-vae-sweep-07-11")
    wandb.agent(sweep_id, start_sweep, count=200)
