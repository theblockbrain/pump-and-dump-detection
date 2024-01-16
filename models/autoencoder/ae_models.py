import tensorflow as tf
from tensorflow.keras.models import Model
from keras.layers import Dense


class Autoencoder_8_6_4(Model):
    def __init__(self, input_units, latent_dimension, activation):
        super().__init__()

        self.encoder = tf.keras.Sequential(
            [
                Dense(input_units, activation=activation, input_shape=[input_units]),
                Dense(6, activation=activation),
                Dense(4, activation=activation),
                Dense(latent_dimension),  # latent dimension
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                Dense(4, activation=activation),
                Dense(6, activation=activation),
                Dense(input_units),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder_8_4(Model):
    def __init__(self, input_units, latent_dimension, activation):
        super().__init__()

        self.encoder = tf.keras.Sequential(
            [
                Dense(input_units, activation=activation, input_shape=[input_units]),
                Dense(4, activation=activation),
                Dense(latent_dimension),  # latent dimension
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                Dense(4, activation=activation),
                Dense(input_units),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder_8_7_5(Model):
    def __init__(self, input_units, latent_dimension, activation):
        super().__init__()

        self.encoder = tf.keras.Sequential(
            [
                Dense(input_units, input_shape=[input_units]),
                Dense(7, activation=activation),
                Dense(5, activation=activation),
                Dense(latent_dimension),  # latent dimension
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                Dense(
                    5,
                    activation=activation,
                ),
                Dense(
                    7,
                    activation=activation,
                ),
                Dense(input_units),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
