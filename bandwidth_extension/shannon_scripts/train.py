import numpy as np
import audio_unet as au

data_dataset_path = '/home/s2142081/summer_project/data.npy'
label_dataset_path = '/home/s2142081/summer_project/label.npy'

X_train = np.load(data_dataset_path)
y_train = np.load(label_dataset_path)
X_train = X_train[:, :, np.newaxis]
y_train = y_train[:, :, np.newaxis]

audio_unet = au.AudioUNet(input_shape=(4096, 1), block_size=6)
audio_unet.show()
audio_unet.compile(learning_rate=5*10e-4)
audio_unet.fit(X_train, y_train, batch_size=16, epochs=200)
audio_unet.save('model_6l.h5')