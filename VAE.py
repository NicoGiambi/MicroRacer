import tensorflow as tf
from tensorflow.keras import layers, regularizers


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    def __init__(self, latent_dim=32, intermediate_dim=64, drop_out_prob=0.1, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.first_dense = layers.Dense(intermediate_dim,
                                        activation="relu",
                                        kernel_regularizer=regularizers.l2(0.0001))
        self.second_dense = layers.Dense(intermediate_dim // 2,
                                         activation="relu",
                                         kernel_regularizer=regularizers.l2(0.0001))
        self.drop_out = layers.Dropout(drop_out_prob)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.first_dense(inputs)
        x = self.drop_out(x)
        x = self.second_dense(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    def __init__(self, original_dim, intermediate_dim=64, drop_out_prob=0.1, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.first_dense = layers.Dense(intermediate_dim // 2,
                                        activation="relu",
                                        kernel_regularizer=regularizers.l2(0.0001))
        self.drop_out = layers.Dropout(drop_out_prob)
        self.second_dense = layers.Dense(intermediate_dim,
                                         activation="relu",
                                         kernel_regularizer=regularizers.l2(0.0001))
        # no activation
        self.dense_output = layers.Dense(original_dim)

    def call(self, inputs):
        x = self.first_dense(inputs)
        x = self.drop_out(x)
        x = self.second_dense(x)
        return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
    def __init__(
            self,
            original_dim,
            intermediate_dim,
            latent_dim,
            drop_out_prob,
            name='autoencoder',
            **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim, drop_out_prob=drop_out_prob)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim, drop_out_prob=drop_out_prob)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed