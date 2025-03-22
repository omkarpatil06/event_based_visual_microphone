# import sys
# import librosa
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import make_scorer, accuracy_score
# sys.path.insert(0, '/Users/omkarpatil/Documents/GitHub/event_based_visual_microphone/utilities/')
# import utility_folders as uf

# def extract_features(file_path):
#     y, sr = librosa.load(file_path)
#     # Temporal Features
#     ste = np.mean(librosa.feature.rms(y=y))
#     ste_variance = np.var(librosa.feature.rms(y=y))
#     # Spectral Features
#     spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
#     spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
#     # Harmonic Features
#     harmonic_ratio = np.mean(librosa.effects.harmonic(y))
#     try:
#         pitch = librosa.core.piptrack(y=y, sr=sr)
#         pitch = np.mean(pitch[pitch > 0])
#     except:
#         pitch = 0  # Sometimes pitch extraction may fail, use 0 as fallback
#     # Rhythmic Features
#     autocorrelation = np.mean(librosa.autocorrelate(y))
#     return [ste, ste_variance, spectral_contrast, spectral_rolloff, harmonic_ratio, pitch, autocorrelation]

# def print_features(file_path):
#     data = extract_features(file_path)
#     print(f"Short-Time Energy: {data[0]}")
#     print(f"Variance of STE: {data[1]}")
#     print(f"Spectral Contrast: {data[2]}")
#     print(f"Spectral Roll-off: {data[3]}")
#     print(f"Harmonic-to-Noise Ratio: {data[4]}")
#     print(f"Pitch: {data[5]}")
#     print(f"Autocorrelation: {data[6]}")


# def make_pandas_dataset(folder_path, save_path):
#     files_paths = uf.get_files(folder_path)
#     features = []
#     for file in files_paths:
#         print(f"Completed with file : {folder_path}{file}")
#         features.append(extract_features(f'{folder_path}{file}'))
#     columns = ['Short-Time Energy', 'STE Variance', 'Spectral Contrast', 'Spectral Roll-off', 'Harmonic Ratio', 'Pitch', 'Autocorrelation']
#     features_df = pd.DataFrame(features, columns=columns)
#     features_df.to_csv(save_path, index=False)

# def custom_score(y_true, y_pred):
#     return accuracy_score(y_true, y_pred)

# def train_model(dataframe_path):
#     features_df = pd.read_csv(dataframe_path)
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(features_df)

#     iso_forest = IsolationForest(contamination=0.1, random_state=42)
#     iso_forest.fit(scaled_features)
#     return iso_forest, scaler
    
# def predict_model(model, scaler, file_path):
#     features = [extract_features(file_path)]
#     columns = ['Short-Time Energy', 'STE Variance', 'Spectral Contrast', 'Spectral Roll-off', 'Harmonic Ratio', 'Pitch', 'Autocorrelation']
#     features_df = pd.DataFrame(features, columns=columns)
#     scaled_features = scaler.transform(features_df)
    
#     anomaly_score = model.predict(scaled_features)
#     if anomaly_score == -1:
#         print("This audio has signal :-)")
#         return True
#     else:
#         print("This audio is noise :-(")
#         return False

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Feature Extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    
    # Temporal Features
    rms = np.mean(librosa.feature.rms(y=y))
    rms_var = np.var(librosa.feature.rms(y=y))
    
    # Frequency-domain Features
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
    
    # Time-domain Features
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    energy_entropy = -np.sum(y**2 * np.log2(y**2 + 1e-12)) / len(y)
    signal_skewness = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    signal_kurtosis = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Combine all features into a single vector
    features = np.array([rms, rms_var, spectral_rolloff, zcr, energy_entropy, signal_skewness, signal_kurtosis])
    features = np.concatenate([features, spectral_contrast])  # Combine with spectral contrast features
    
    return features

# Dataset Creation
def create_dataset(noise_folder, noisy_audio_folder, save_path):
    noise_files = [os.path.join(noise_folder, file) for file in os.listdir(noise_folder) if file.endswith('.wav')]
    noisy_audio_files = [os.path.join(noisy_audio_folder, file) for file in os.listdir(noisy_audio_folder) if file.endswith('.wav')]
    
    features = []
    labels = []
    
    # Define feature names including spectral contrast bands
    feature_names = ['RMS', 'RMS Variance', 'Spectral Rolloff', 'ZCR', 'Energy Entropy', 'Skewness', 'Kurtosis'] + \
                    [f'Spectral Contrast {i}' for i in range(7)]
    
    for file in noise_files:
        features.append(extract_features(file))
        labels.append(0)  # Label 0 for pure noise
    
    for file in noisy_audio_files:
        features.append(extract_features(file))
        labels.append(1)  # Label 1 for noisy audio with signal
    
    features_df = pd.DataFrame(features, columns=feature_names)
    features_df['Label'] = labels
    features_df.to_csv(save_path, index=False)
    print(f"Dataset saved to {save_path}")

# Model Training
def train_model(dataframe_path):
    features_df = pd.read_csv(dataframe_path)
    X = features_df.drop(columns=['Label'])  # Features
    y = features_df['Label']  # Labels
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Preserve column names
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    gbm_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    gbm_model.fit(X_train, y_train)

    y_pred = gbm_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    return gbm_model, scaler

# Prediction
def predict_model(model, scaler, file_path):
    # Extract features using the defined feature extraction function
    features = extract_features(file_path)
    
    # Define feature names (Ensure these match the model's training features)
    feature_names = ['RMS', 'RMS Variance', 'Spectral Rolloff', 'ZCR', 'Energy Entropy', 'Skewness', 'Kurtosis'] + \
                    ['Spectral Contrast ' + str(i) for i in range(7)]  # Adjust based on the actual number of spectral contrast bands used
    
    # Create DataFrame with correct feature names
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # Scale features using the pre-fitted scaler
    scaled_features = scaler.transform(features_df)
    
    # Make predictions using the pre-trained model
    prediction = model.predict(scaled_features)
    
    # Interpret and print the prediction result
    if prediction[0] == 1:
        print("This audio has a signal :-)")
        return True
    else:
        print("This audio is noise :-(")
        return False

# Searching for signals in a folder
def search_folder_for_signals(folder_path, model, scaler):
    # List all WAV files in the folder
    audio_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.wav')]
    results = []
    
    # Define feature names based on the model's input expectations
    feature_names = ['RMS', 'RMS Variance', 'Spectral Rolloff', 'ZCR', 'Energy Entropy', 'Skewness', 'Kurtosis'] + \
                    ['Spectral Contrast ' + str(i) for i in range(7)]  # Adjust based on the actual number of spectral contrast bands used
    
    for file in audio_files:
        # Extract features for each file
        features = extract_features(file)
        
        # Create DataFrame with correct feature names
        features_df = pd.DataFrame([features], columns=feature_names)
        
        # Scale features using the pre-fitted scaler
        scaled_features = scaler.transform(features_df)
        
        # Make predictions using the pre-trained model
        prediction = model.predict(scaled_features)
        
        # Append results
        if prediction[0] == 1:
            results.append((file, "Signal Detected"))
            print(f"{file}: Signal Detected :-)")
        else:
            results.append((file, "No Signal Detected"))
            print(f"{file}: No Signal Detected :-(")

    return results
