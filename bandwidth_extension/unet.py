import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

class UNet:
    def __init__(self, input_shape):
        self.unet_model = None
        self.kl_loss = None
        self.build_unet(input_shape=input_shape)

    ##################### TRAINING MODEL #####################

    # UNet loss at each gradient step
    def combined_loss(self, y_true, y_pred):
        mse_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
        return mse_loss

    # Compiling the UNet
    def compile(self, learning_rate=5e-4):
        optimiser = Adam(learning_rate)
        self.unet_model.compile(optimizer=optimiser, loss=self.combined_loss)
    
    # Training and saving the UNet
    def fit(self, X_train, y_train, batch_size, epochs, save_path):
        self.unet_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
        self.unet_model.save_weights(save_path)

    # Using the model to predict on new data
    def predict(self, X_test):
        y_pred = self.unet_model.predict(X_test)
        return y_pred
    
    # Loading the weights of UNet
    def load_weights(self, weights_path):
        self.unet_model.load_weights(weights_path)

    @classmethod
    def load(cls, input_shape, weights_path):
        unet = UNet(input_shape=input_shape)
        unet.unet_model.load_weights(weights_path)
        return unet

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
    
    # Bottleneck of the unet
    def bottleneck_layer(self, input, num_filters, kernel_size, strides, padding):
        conv2d_output = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu')(input)
        batchnorm_output = layers.BatchNormalization()(conv2d_output)
        return batchnorm_output
    
    # Create the unet model
    def build_unet(self, input_shape):
        # VUNet encoder
        input_layer = layers.Input(shape=input_shape)
        first_downsampling = self.encoder_layer(input_layer, 16, 3, 2, 'same')
        second_downsampling = self.encoder_layer(first_downsampling, 32, 5, 2, 'same')
        third_downsampling = self.encoder_layer(second_downsampling, 64, 5, 2, 'same')
        fourth_downsampling = self.encoder_layer(third_downsampling, 64, 5, 2, 'same')
        fifth_downsampling = self.encoder_layer(fourth_downsampling, 64, 3, 2, 'same')
        # VUNet bottleneck
        bottleneck_layer = self.bottleneck_layer(fifth_downsampling, 64, 3, 2, 'same')
        # VUNet decoder
        fifth_upsampling = self.decoder_layer(bottleneck_layer, 64, 3, 2, 'same')
        fifth_concatenate = layers.concatenate([fifth_downsampling, fifth_upsampling])
        fourth_upsampling = self.decoder_layer(fifth_concatenate, 64, 5, 2, 'same')
        fourth_concatenate = layers.concatenate([fourth_downsampling, fourth_upsampling])
        third_upsampling = self.decoder_layer(fourth_concatenate, 64, 5, 2, 'same')
        third_concatenate = layers.concatenate([third_downsampling, third_upsampling])
        second_upsampling = self.decoder_layer(third_concatenate, 32, 5, 2, 'same')
        second_concatenate = layers.concatenate([second_downsampling, second_upsampling])
        first_upsampling  = self.decoder_layer(second_concatenate, 16, 3, 2, 'same')
        first_concatenate = layers.concatenate([first_downsampling, first_upsampling])
        output_upsampling = layers.Conv2DTranspose(1, 3, 2, 'same')(first_concatenate)
        output_layer = layers.Activation('sigmoid')(output_upsampling)
        self.unet_model = Model(input_layer, output_layer, name='VUNetModel')