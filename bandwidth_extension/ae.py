import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

class AE:
    def __init__(self, input_shape):
        self.ae_model = None
        self.kl_loss = None
        self.build_autoencoder(input_shape=input_shape)

    ##################### TRAINING MODEL #####################

    # AE loss at each gradient step
    def combined_loss(self, y_true, y_pred):
        mse_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
        return mse_loss

    # Compiling the AE
    def compile(self, learning_rate=5e-4):
        optimiser = Adam(learning_rate)
        self.ae_model.compile(optimizer=optimiser, loss=self.combined_loss)
    
    # Training and saving the AE
    def fit(self, X_train, y_train, batch_size, epochs, save_path):
        self.ae_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
        self.ae_model.save_weights(save_path)

    # Using the model to predict on new data
    def predict(self, X_test):
        y_pred = self.ae_model.predict(X_test)
        return y_pred
    
    # Loading the weights of AE
    def load_weights(self, weights_path):
        self.ae_model.load_weights(weights_path)

    @classmethod
    def load(cls, input_shape, weights_path):
        vae = AE(input_shape=input_shape)
        vae.ae_model.load_weights(weights_path)
        return vae

    ##################### CONSTRUCTING MODEL #####################

    # Downsampling layer of the encoder
    def encoder_layer(self, input, num_filters, kernel_size, strides, padding):
        conv2d_output = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu')(input)
        batchnorm_output = layers.BatchNormalization()(conv2d_output)
        return batchnorm_output

    # Upsampling layer of the decoder
    def decoder_layer(self, input, filters, kernel_size, strides, padding):
        conv2d_transpose = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu')(input)
        batchnorm_output = layers.BatchNormalization()(conv2d_transpose)
        return batchnorm_output  
    
    # Bottleneck of the autoencoder
    def bottleneck_layer(self, input, num_filters, kernel_size, strides, padding):
        conv2d_output = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu')(input)
        batchnorm_output = layers.BatchNormalization()(conv2d_output)
        return batchnorm_output
    
    # Create the autoencoder model
    def build_autoencoder(self, input_shape):
        # VAE encoder
        input_layer = layers.Input(shape=input_shape)
        first_downsampling = self.encoder_layer(input_layer, 16, 3, 2, 'same')
        second_downsampling = self.encoder_layer(first_downsampling, 32, 5, 2, 'same')
        third_downsampling = self.encoder_layer(second_downsampling, 64, 5, 2, 'same')
        fourth_downsampling = self.encoder_layer(third_downsampling, 64, 5, 2, 'same')
        fifth_downsampling = self.encoder_layer(fourth_downsampling, 64, 3, 2, 'same')
        # VAE bottleneck
        bottleneck_layer = self.bottleneck_layer(fifth_downsampling, 64, 3, 2, 'same')
        # VAE decoder
        fifth_upsampling = self.decoder_layer(bottleneck_layer, 64, 3, 2, 'same')
        fourth_upsampling = self.decoder_layer(fifth_upsampling, 64, 5, 2, 'same')
        third_upsampling = self.decoder_layer(fourth_upsampling, 64, 5, 2, 'same')
        second_upsampling = self.decoder_layer(third_upsampling, 32, 5, 2, 'same')
        first_upsampling  = self.decoder_layer(second_upsampling, 16, 3, 2, 'same')
        output_upsampling = layers.Conv2DTranspose(1, 3, 2, 'same')(first_upsampling)
        output_layer = layers.Activation('sigmoid')(output_upsampling)
        self.ae_model = Model(input_layer, output_layer, name='VAEModel')
        # Display model summary
        self.ae_model.summary()