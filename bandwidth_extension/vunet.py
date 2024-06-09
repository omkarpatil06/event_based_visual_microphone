import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses

# Custom Sampling layer for VAE
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def get_config(self):
        return super().get_config()

# Main VAE model class
class VAE(Model):
    """Builds a Variational Autoencoder."""
    def __init__(self, input_shape, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.encoder = self.build_encoder(input_shape)
        self.optimizer = optimizers.Adam()  # Optimizer for the model
        # Trackers for training
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    
    def get_config(self):
        config = super().get_config()
        config.update({'input_shape': self.input_shape})
        return config

    def call(self, inputs):
        """Executes the model on new inputs and returns outputs."""
        z_mean, z_log_var, reconstructed = self.encoder(inputs)
        return reconstructed, z_mean, z_log_var
    
    def train_step(self, data):
        """Training logic for one step."""
        with tf.GradientTape() as tape:
            reconstructed, z_mean, z_log_var = self(data, training=True)
            reconstruction_loss = tf.reduce_mean(losses.mean_squared_error(data, reconstructed))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }
        
    # Helper methods to build encoder and decoder
    def encoder_layer(self, input, num_filters, kernel_size, strides, padding):
        """Creates a convolutional block for the encoder."""
        conv2d_output = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu')(input)
        batchnorm_output = layers.BatchNormalization()(conv2d_output)
        return batchnorm_output

    def decoder_layer(self, input, filters, kernel_size, strides, padding):
        """Creates a convolutional block for the decoder."""
        conv2d_transpose = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu')(input)
        batchnorm_output = layers.BatchNormalization()(conv2d_transpose)
        return batchnorm_output  

    def build_encoder(self, input_shape):
        """Constructs the encoder part of VAE."""
        input_layer = layers.Input(shape=input_shape)
        x1 = self.encoder_layer(input_layer, 16, 3, 2, 'same')
        x2 = self.encoder_layer(x1, 32, 5, 2, 'same')
        x3 = self.encoder_layer(x2, 64, 5, 2, 'same')
        x4 = self.encoder_layer(x3, 64, 5, 2, 'same')
        x5 = self.encoder_layer(x4, 64, 3, 2, 'same')
        x = layers.Conv2D(64, 3, 2, 'same', activation='relu')(x5)
        x = layers.Flatten()(x)
        z_mean = layers.Dense(32, name='z_mean')(x)
        z_log_var = layers.Dense(32, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        x = layers.Dense(64 * 4 * 4)(z)
        x = layers.Reshape((4, 4, 64))(x)
        x = self.decoder_layer(x, 64, 3, 2, 'same')
        u = layers.concatenate([x5, x])
        x = self.decoder_layer(u, 64, 5, 2, 'same')
        u = layers.concatenate([x4, x])
        x = self.decoder_layer(u, 64, 5, 2, 'same')
        u = layers.concatenate([x3, x])
        x = self.decoder_layer(u, 32, 5, 2, 'same')
        u = layers.concatenate([x2, x])
        x = self.decoder_layer(u, 16, 3, 2, 'same')
        u = layers.concatenate([x1, x])
        output_upsampling = layers.Conv2DTranspose(1, 3, 2, 'same')(u)
        output_layer = layers.Activation('sigmoid')(output_upsampling)
        return Model(input_layer, [z_mean, z_log_var, output_layer], name='vunet')