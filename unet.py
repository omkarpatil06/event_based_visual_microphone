import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dropout, Activation, LeakyReLU, Concatenate, Add
from tensorflow.keras.initializers import Orthogonal, RandomNormal
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer

class SubPixel1D(Layer):
    def __init__(self, r, **kwargs):
        super(SubPixel1D, self).__init__(**kwargs)
        self.r = r

    def call(self, inputs):
        with tf.compat.v1.name_scope('subpixel'):
            X = tf.transpose(inputs, perm=[2, 1, 0])  # (r, w, b)
            X = tf.batch_to_space(X, [self.r], [[0, 0]])  # (1, r*w, b)
            X = tf.transpose(X, perm=[2, 1, 0])
            return X

    def compute_output_shape(self, input_shape):
        new_width = input_shape[1] * self.r
        return (input_shape[0], new_width, input_shape[2] // self.r)

class UNet:
    def __init__(self):
        self.dropout_rate = 0.3
        self.unet_model = self.build_model()
    
    def combined_loss(self, y_true, y_pred):
        mse_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
        return mse_loss

    # Compiling the UNet
    def compile(self, learning_rate=5e-4):
        optimiser = Adam(learning_rate)
        self.unet_model.compile(optimizer=optimiser, loss='mean_squared_error')
    
    # Training and saving the UNet
    def fit(self, X_train, y_train, batch_size, epochs, save_path):
        self.unet_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
        self.unet_model.save_weights(save_path)

    # Using the model to predict on new data
    def predict(self, X_test):
        y_pred = self.unet_model.predict(X_test)
        return y_pred
    
    # Loading the weights of UNet
    def load_weights(self, weights_path):
        self.unet_model.load_weights(weights_path)

    @classmethod
    def load(cls, weights_path):
        unet = UNet()
        unet.unet_model.load_weights(weights_path)
        return unet

    def build_model(self):
        downsampling_layers = []
        # Layer 0
        X = Input(shape=(512, 1))
        x = Conv1D(filters=128, kernel_size=65, activation=None, padding='same', strides=2)(X)
        x = LeakyReLU(0.2)(x)
        downsampling_layers.append(x)
        # Layer 1
        x = Conv1D(filters=384, kernel_size=33, activation=None, padding='same', strides=2)(x)
        x = LeakyReLU(0.2)(x)
        downsampling_layers.append(x)
        # Layer 2
        x = Conv1D(filters=512, kernel_size=17, activation=None, padding='same', strides=2)(x)
        x = LeakyReLU(0.2)(x)
        downsampling_layers.append(x)
        # Layer 3
        x = Conv1D(filters=512, kernel_size=9, activation=None, padding='same', strides=2)(x)
        x = LeakyReLU(0.2)(x)
        downsampling_layers.append(x)
        # Layer 4
        x = Conv1D(filters=512, kernel_size=9, activation=None, padding='same', strides=2)(x)
        x = LeakyReLU(0.2)(x)
        downsampling_layers.append(x)
        # Layer 5
        x = Conv1D(filters=512, kernel_size=9, activation=None, padding='same', strides=2)(x)
        x = LeakyReLU(0.2)(x)
        downsampling_layers.append(x)
        # Layer 6
        x = Conv1D(filters=512, kernel_size=9, activation=None, padding='same', strides=2)(x)
        x = LeakyReLU(0.2)(x)
        downsampling_layers.append(x)
        # Layer 7
        x = Conv1D(filters=512, kernel_size=9, activation=None, padding='same', strides=2)(x)
        x = LeakyReLU(0.2)(x)
        downsampling_layers.append(x)
        # Layer 8
        x = Conv1D(filters=512, kernel_size=9, activation=None, padding='same', strides=2)(x)
        x = LeakyReLU(0.2)(x)
        downsampling_layers.append(x)
        # Bottleneck layer
        x = Conv1D(filters=512, kernel_size=9, activation=None, padding='same', strides=2)(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = LeakyReLU(0.2)(x)
        # Layer 8
        x = Conv1D(filters=1024, kernel_size=9, activation=None, padding='same')(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = Activation('relu')(x)
        x = SubPixel1D(r=2)(x)
        x = Concatenate(axis=2)([x, downsampling_layers[7]]) 
        # Layer 7
        x = Conv1D(filters=1024, kernel_size=9, activation=None, padding='same')(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = Activation('relu')(x)
        x = SubPixel1D(r=2)(x)
        x = Concatenate(axis=2)([x, downsampling_layers[6]])
        # Layer 6
        x = Conv1D(filters=1024, kernel_size=9, activation=None, padding='same')(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = Activation('relu')(x)
        x = SubPixel1D(r=2)(x)
        x = Concatenate(axis=2)([x, downsampling_layers[5]])
        # Layer 5
        x = Conv1D(filters=1024, kernel_size=9, activation=None, padding='same')(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = Activation('relu')(x)
        x = SubPixel1D(r=2)(x)
        x = Concatenate(axis=2)([x, downsampling_layers[4]])
        # Layer 4
        x = Conv1D(filters=1024, kernel_size=9, activation=None, padding='same')(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = Activation('relu')(x)
        x = SubPixel1D(r=2)(x)
        x = Concatenate(axis=2)([x, downsampling_layers[3]])
        # Layer 3
        x = Conv1D(filters=1024, kernel_size=9, activation=None, padding='same')(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = Activation('relu')(x)
        x = SubPixel1D(r=2)(x)
        x = Concatenate(axis=2)([x, downsampling_layers[2]])
        # Layer 2
        x = Conv1D(filters=768, kernel_size=33, activation=None, padding='same')(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = Activation('relu')(x)
        x = SubPixel1D(r=2)(x)
        x = Concatenate(axis=2)([x, downsampling_layers[1]])
        # Layer 1
        x = Conv1D(filters=256, kernel_size=65, activation=None, padding='same')(x)
        x = Dropout(rate=self.dropout_rate)(x)
        x = Activation('relu')(x)
        x = SubPixel1D(r=2)(x)
        x = Concatenate(axis=2)([x, downsampling_layers[0]])
        # Final convolution layer
        x = Conv1D(filters=2, kernel_size=9, activation=None, padding='same')(x)
        x = SubPixel1D(r=2)(x)
        g = Add()([x, X])
        model = Model(X, g, name='UNet')
        model.summary()
        return model