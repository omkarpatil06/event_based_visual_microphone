import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

##########################################################
# SUBPIXELSHIFT CODE
##########################################################
class SubPixelShift(layers.Layer):
    def __init__(self, **kwargs):
        super(SubPixelShift, self).__init__(**kwargs)

    @tf.autograph.experimental.do_not_convert
    def call(self, x, **kwargs):
        X = tf.transpose(a=x, perm=[2,1,0]) # (r, w, b)
        X = tf.batch_to_space(X, [2], [[0,0]]) # (1, r*w, b)
        X = tf.transpose(a=X, perm=[2,1,0])
        return X

##########################################################
# AUDIO UNET MODEL CODE
##########################################################
class AudioUNet:
    def __init__(self, input_shape=(4096, 1), block_size=5):
        self.input_shape = input_shape
        self.block_size = block_size

        self.encoder_conv_filters = []
        self.encoder_conv_lengths = []
        self.bottleneck_conv_filters = None
        self.bottleneck_conv_lengths = None
        self.decoder_conv_filters = []
        self.decoder_conv_lengths = []
        self.audio_unet_stats()

        self.encoder_layers = []
        self.decoder_layers = []
        self.audio_unet = self.build_audio_unet()

    ##########################################################
    # TRAINING MODEL
    ##########################################################
    def show(self):
        self.audio_unet.summary()

    def compile(self, learning_rate):
        self.audio_unet.compile(optimizer=Adam(learning_rate), loss='mean_squared_error', metrics=[self.snr, self.lsd])

    def fit(self, X_train, y_train, batch_size, epochs):
        self.audio_unet.fit(X_train, y_train, batch_size= batch_size, epochs=epochs, shuffle=True)
    
    def save(self, save_path):
        self.audio_unet.save(save_path)
    
    def predict(self, X_test):
        return self.audio_unet.predict(X_test)

    @classmethod
    def load(cls, weights_path):
        unet = cls()
        unet.audio_unet.load_weights(weights_path)
        return unet

    def snr(self, y_true, y_pred):
        epsilon = 1e-10 
        signal_power = tf.reduce_sum(tf.square(y_true), axis=-1) + epsilon
        noise_power = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1) + epsilon
        snr_ratio = signal_power / noise_power
        snr_db = 10.0 * tf.math.log(snr_ratio) / tf.math.log(10.0)
        return tf.reduce_mean(snr_db)
    
    def lsd(self, y_true, y_pred, L=1, K=1):
        def power_spectrum(x):
            return tf.square(tf.abs(tf.signal.stft(x, frame_length=L, frame_step=K)))
        
        ps_true = power_spectrum(y_true)
        ps_pred = power_spectrum(y_pred)
        
        lsd_value = tf.sqrt(tf.reduce_mean(tf.square(tf.math.log(ps_true + 1e-10) - tf.math.log(ps_pred + 1e-10)), axis=-1))
        return tf.reduce_mean(lsd_value)

    ##########################################################
    # BUILDING MODEL
    ##########################################################
    def audio_unet_stats(self):
        for idx in range(self.block_size):
            bidx = idx+1
            num_filters = max(2**(6+bidx), 512)
            filter_length = min(2**(7-bidx)+1, 9)
            self.encoder_conv_filters.append(num_filters)
            self.encoder_conv_lengths.append(filter_length)

            num_filters = max(2**(7+(self.block_size-bidx+1)), 1024)
            filter_length = min(2**(7-(self.block_size-bidx+1))+1, 9)
            self.decoder_conv_filters.append(num_filters)
            self.decoder_conv_lengths.append(filter_length)
        self.bottleneck_conv_filters = max(2**(6+self.block_size+1), 512)
        self.bottleneck_conv_lengths = min(2**(7-self.block_size+1)+1, 9)

    def build_audio_unet(self):
        self.encoder()
        self.bottleneck()
        self.decoder()
        return Model(self.encoder_layers[0], self.decoder_layers[-1], name='Network')

    def encoder(self):
        self.encoder_layers = [layers.Input(shape=self.input_shape)]
        for nfilt, flen in zip(self.encoder_conv_filters, self.encoder_conv_lengths):
            x = self.encoder_layers[-1]
            x = layers.Conv1D(filters=nfilt, kernel_size=flen, strides=2, padding='same')(x)
            x = layers.LeakyReLU(0.2)(x)
            self.encoder_layers.append(x)

    def bottleneck(self):
        x = self.encoder_layers[-1]
        x = layers.Conv1D(filters=self.bottleneck_conv_filters, kernel_size=self.bottleneck_conv_lengths, strides=2, padding='same')(x)
        x = layers.Dropout(rate=0.5)(x)
        self.decoder_layers.append(x)

    def decoder(self):
        skip_connections = self.encoder_layers[::-1]
        for nfilt, flen, idx in zip(self.decoder_conv_filters, self.decoder_conv_lengths, range(len(skip_connections))):
            x = self.decoder_layers[-1]
            x = layers.Conv1D(filters=nfilt, kernel_size=flen, padding='same')(x)
            x = layers.Dropout(rate=0.5)(x)
            x = layers.ReLU()(x)
            x = SubPixelShift()(x)
            x = layers.Concatenate(axis=-1)([x, skip_connections[idx]])
            self.decoder_layers.append(x)
        x = layers.Conv1D(filters=2, kernel_size=9, padding='same')(x)
        x = SubPixelShift()(x)
        self.decoder_layers.append(x)
        x = layers.Add()([x, self.encoder_layers[0]])
        self.decoder_layers.append(x)
    