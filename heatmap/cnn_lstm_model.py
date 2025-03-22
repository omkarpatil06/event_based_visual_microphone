import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, TimeDistributed, Dense, Flatten

def load_video(video_path, num_frames=512, resize_shape=(30, 30)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = np.linspace(0, total_frames - 1, num=num_frames).astype(int)
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_idx:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(frame, resize_shape)
            frames.append(resized_frame)
    cap.release()
    frames = np.array(frames)  
    frames = np.expand_dims(frames, axis=-1)    # shape: num_frames, 30, 30, 1
    frames = frames / 255.0

def load_dataset(video_folder, num_frames=512):
    videos = []
    labels = []

    for video_file in os.listdir(video_folder):
        if video_file.endswith('.mp4'):
            label = 0 if 'signal' in video_file else 1
            video_path = os.path.join(video_folder, video_file)
            video_data = load_video(video_path, num_frames=num_frames)
            videos.append(video_data)
            labels.append(label)
    return np.array(videos), np.array(labels)

def create_cnn_lstm_model(input_shape=(512, 30, 30, 1), num_classes=2):
    num_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)
    model = Sequential()
    
    # TimeDistributed Conv2D to apply the CNN on each frame
    model.add(TimeDistributed(Conv2D(num_filters, kernel_size, activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size = pool_size)))
    model.add(TimeDistributed(Conv2D(2*num_filters, kernel_size, activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size = pool_size)))
    model.add(TimeDistributed(Flatten()))

    # LSTM to capture temporal dependencies
    model.add(LSTM(128, return_sequences=False))

    # Fully connected layer for classification
    model.add(Dense(2*num_filters, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model