import os
import numpy as np
from unet import UNet
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X_dataset = '/home/s2142081/unet_train/data.npy'
y_dataset = '/home/s2142081/unet_train/label.npy'
X, y = np.load(X_dataset), np.load(y_dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

epochs = 100
batch_size = 32

model = UNet()
model.compile()
model.fit(X_train, y_train, batch_size, epochs, 'model_weights.h5')