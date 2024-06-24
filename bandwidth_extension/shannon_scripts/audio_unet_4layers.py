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
    def __init__(self, input_shape=(4096, 1), block_size=4):
        self.input_shape = input_shape
        self.block_size = block_size

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
    def build_audio_unet(self):
        # INPUT LAYER
        input = layers.Input(shape=self.input_shape)
        
        # ENCODER
        e1 = layers.Conv1D(filters=512, kernel_size=9, strides=2, padding='same')(input)
        e1 = layers.LeakyReLU(0.2)(e1)
        e2 = layers.Conv1D(filters=512, kernel_size=9, strides=2, padding='same')(e1)
        e2 = layers.LeakyReLU(0.2)(e2)
        e3 = layers.Conv1D(filters=512, kernel_size=9, strides=2, padding='same')(e2)
        e3 = layers.LeakyReLU(0.2)(e3)
        e4 = layers.Conv1D(filters=1024, kernel_size=9, strides=2, padding='same')(e3)
        e4 = layers.LeakyReLU(0.2)(e4)

        # BOTTLENECK
        b = layers.Conv1D(filters=2048, kernel_size=5, strides=2, padding='same')(e4)
        b = layers.Dropout(rate=0.5)(b)

        # DECODER
        d1 = layers.Conv1D(filters=2048, kernel_size=9, padding='same')(b)
        d1 = layers.Dropout(rate=0.5)(d1)
        d1 = layers.ReLU()(d1)
        d1 = SubPixelShift()(d1)
        d1 = layers.Concatenate(axis=-1)([d1, e4])
        d2 = layers.Conv1D(filters=1024, kernel_size=9, padding='same')(d1)
        d2 = layers.Dropout(rate=0.5)(d2)
        d2 = layers.ReLU()(d2)
        d2 = SubPixelShift()(d2)
        d2 = layers.Concatenate(axis=-1)([d2, e3])
        d3 = layers.Conv1D(filters=1024, kernel_size=9, padding='same')(d2)
        d3 = layers.Dropout(rate=0.5)(d3)
        d3 = layers.ReLU()(d3)
        d3 = SubPixelShift()(d3)
        d3 = layers.Concatenate(axis=-1)([d3, e2])
        d4 = layers.Conv1D(filters=1024, kernel_size=9, padding='same')(d3)
        d4 = layers.Dropout(rate=0.5)(d4)
        d4 = layers.ReLU()(d4)
        d4 = SubPixelShift()(d4)
        d4 = layers.Concatenate(axis=-1)([d4, e1])
        
        # OUTPUT LAYER
        output = layers.Conv1D(filters=2, kernel_size=9, padding='same')(d4)
        output = SubPixelShift()(output)
        output = layers.Add()([output, input])
        return Model(input, output, name='Network')