import os
import shutil
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.signal import correlate
import librosa.display
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.fftpack import fft

# UTILITY SIGNALS:
def show_signal(signal, signal_title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(signal)
    ax.set_xlabel('Number of points [n]')
    ax.set_ylabel('Amplitude [A]')
    ax.set_title(f'Time Series of {signal_title} Signal')
    ax.grid()
    plt.show()

# UTILITY STFT FUNCTIONS:
def signal_stft(signal, n_fft, hop_length, window=windows.blackmanharris):
    stft_matrix = librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length, window=window)
    mag = np.abs(stft_matrix)
    return mag

# UTILITY ISTFT FUNCTIONS:
def signal_istft(mag, phase, n_fft, hop_length, window=windows.blackmanharris):
    stft_matrix = mag*np.exp(1j*phase)
    signal = librosa.istft(stft_matrix=stft_matrix, n_fft=n_fft, hop_length=hop_length, window=window)
    return signal

# UTILITY SHOW SPECTROGRAM:
def show_spectrogram(signal, sr, n_fft, hop_length, signal_title):
    mag_matrix = signal_stft(signal, n_fft, hop_length)
    fig, ax = plt.subplots(figsize=(4, 4))
    image = librosa.display.specshow(data=mag_matrix, sr=sr, n_fft=n_fft, vmin=-4.5, vmax=4.5, hop_length=hop_length, y_axis='log', x_axis='time', cmap='jet', ax=ax)
    ax.set_title(f'Magnitude Spectrogram of {signal_title} Signal')
    fig.colorbar(image, ax=ax, format='%+2.f dB')
    plt.show()

def show_spectrograms_in_folder(folder_path, sr, n_fft, hop_length):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            signal, sr = librosa.load(file_path, sr=None)
            print(f"Displaying time series plot and spectrogram for {file_name}...")
            show_spectrogram(signal, sr, n_fft, hop_length, file_name)
    
# UTILITY SEGMENTAL SNR:
def segmental_snr(original, reconstructed, segment_length=512, overlap=0.5):
    step = int(segment_length * (1 - overlap))
    seg_snr = []
    # Ensure that both signals are numpy arrays
    original = np.array(original)
    reconstructed = np.array(reconstructed)
    # Calculate SNR for each segment
    for start in range(0, len(original) - segment_length + 1, step):
        end = start + segment_length
        orig_segment = original[start:end]
        rec_segment = reconstructed[start:end]
        # Calculate the noise segment
        noise = orig_segment - rec_segment
        # Calculate signal energy and noise energy
        signal_energy = np.sum(orig_segment ** 2)
        noise_energy = np.sum(noise ** 2)
        # Avoid division by zero
        if noise_energy == 0:
            noise_energy = 1e-10
        # Calculate SNR for this segment
        snr = 10 * np.log10(signal_energy / noise_energy)
        seg_snr.append(snr)
    snr_value = np.mean(seg_snr)
    print("Segmental SNR: {:.2f} dB".format(snr_value))

# Function to compute the spectrogram of a signal
def compute_spectrogram(signal, sr, n_fft=256, hop_length=128):
    S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
    return librosa.amplitude_to_db(S, ref=np.max)

# Function to compute the Log-Spectral Distance
def log_spectral_distance(S1, S2):
    return np.mean(np.abs(S1 - S2))

# Function to compute Cross-Correlation
def cross_correlation(S1, S2):
    return np.max(correlate(S1.flatten(), S2.flatten()))

# Function to compute Dynamic Time Warping (DTW) distance
def dtw_distance(S1, S2):
    distance, _ = fastdtw(S1.T, S2.T, dist=euclidean)
    return distance

# Load the reference signal
def load_audio(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    return signal, sr

def compare_spectrograms_metrics(folder_path, reference_file):
    ref_signal, ref_sr = load_audio(reference_file)
    ref_spectrogram = compute_spectrogram(ref_signal, ref_sr)
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav') and file_name != os.path.basename(reference_file):
            file_path = os.path.join(folder_path, file_name)
            signal, sr = load_audio(file_path)
            spectrogram = compute_spectrogram(signal, sr)
            lsd = log_spectral_distance(ref_spectrogram, spectrogram)
            cc = cross_correlation(ref_spectrogram, spectrogram)
            dtw_dist = dtw_distance(ref_spectrogram, spectrogram)
            results.append({
                'file': file_name,
                'log_spectral_distance': lsd,
                'cross_correlation': cc,
                'dtw_distance': dtw_dist})
    ranked_results = sorted(results, key=lambda x: (x['log_spectral_distance'], x['dtw_distance'], -x['cross_correlation']))
    print("\nRanked Results:")
    for idx, result in enumerate(ranked_results, 1):
        print(f"Rank {idx}: {result['file']}")
        print(f"  Log-Spectral Distance: {result['log_spectral_distance']}")
        print(f"  Cross-Correlation: {result['cross_correlation']}")
        print(f"  DTW Distance: {result['dtw_distance']}\n")

def compare_signals(reference_file, folder_path):
    ref_signal, ref_sr = load_audio(reference_file)
    num_files = len([f for f in os.listdir(folder_path) if f.endswith('.wav')]) + 1  # +1 for the reference
    fig, axes = plt.subplots(num_files, 1, figsize=(10, 2*num_files), sharex=True)
    axes[0].plot(np.arange(len(ref_signal)) / ref_sr, ref_signal, color='blue')
    axes[0].set_title(f"Reference: {os.path.basename(reference_file)}")
    axes[0].set_ylabel("Amplitude")
    plot_idx = 1
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav') and file_name != os.path.basename(reference_file):
            file_path = os.path.join(folder_path, file_name)
            signal, sr = load_audio(file_path)
            axes[plot_idx].plot(np.arange(len(signal)) / sr, signal, color='orange')
            axes[plot_idx].set_title(f"Compared Signal: {file_name}")
            axes[plot_idx].set_ylabel("Amplitude")
            plot_idx += 1
    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()

def compare_spectrograms(reference_file, folder_path):
    ref_signal, ref_sr = load_audio(reference_file)
    ref_spectrogram = compute_spectrogram(ref_signal, ref_sr)
    num_files = len([f for f in os.listdir(folder_path) if f.endswith('.wav')]) + 1  # +1 for the reference
    fig, axes = plt.subplots(num_files, 1, figsize=(10, 3*num_files), sharex=True)
    img = librosa.display.specshow(ref_spectrogram, sr=ref_sr, hop_length=128, x_axis='time', y_axis='log', ax=axes[0], cmap='jet', vmin=-25, vmax=0)
    axes[0].set_title(f"Reference Spectrogram: {os.path.basename(reference_file)}")
    fig.colorbar(img, ax=axes[0], format="%+2.f dB")
    plot_idx = 1
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav') and file_name != os.path.basename(reference_file):
            file_path = os.path.join(folder_path, file_name)
            signal, sr = load_audio(file_path)
            spectrogram = compute_spectrogram(signal, sr)
            img = librosa.display.specshow(spectrogram, sr=sr, hop_length=128, x_axis='time', y_axis='log', ax=axes[plot_idx], cmap='jet', vmin=-25, vmax=0)
            axes[plot_idx].set_title(f"Spectrogram: {file_name}")
            fig.colorbar(img, ax=axes[plot_idx], format="%+2.f dB")
            plot_idx += 1
    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()

def process_folder(reference_file, folder_path):
    ref_signal, ref_sr = load_audio(reference_file)
    ref_spectrogram = compute_spectrogram(ref_signal, ref_sr)
    results = []
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    if not wav_files:
        print(f"No .wav files found in {folder_path}")
        return []
    for file_name in wav_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing file: {file_name}")
        signal, sr = load_audio(file_path)
        spectrogram = compute_spectrogram(signal, sr)
        lsd = log_spectral_distance(ref_spectrogram, spectrogram)
        cc = cross_correlation(ref_spectrogram, spectrogram)
        dtw_dist = dtw_distance(ref_spectrogram, spectrogram)
        results.append({
            'file': file_name.split('.')[0],
            'log_spectral_distance': lsd,
            'cross_correlation': cc,
            'dtw_distance': dtw_dist})
    if not results:
        print(f"No valid comparisons for {folder_path}")
        return []
    ranked_results = sorted(results, key=lambda x: (x['log_spectral_distance'], x['dtw_distance'], -x['cross_correlation']))
    if len(ranked_results) < 2:
        return ranked_results
    return ranked_results[-2:]

def compare_spectrograms_across_videos(folder_path, reference_file):
    worst_files = []  
    if os.path.isdir(folder_path):
        print(f"Processing {folder_path}...")
        worst_in_folder = process_folder(reference_file, folder_path)
        if worst_in_folder: 
            worst_files.append([f['file'] for f in worst_in_folder])
        else:
            print(f"No valid comparisons found for {folder_path}")
    return worst_files

def copy_files_to_folder(file_paths, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for file_path in file_paths:
        if os.path.isfile(file_path):  # Ensure it's a valid file path
            file_name = os.path.basename(file_path)  # Extract the file name
            dest_path = os.path.join(destination_folder, file_name)  # Create the destination path
            shutil.copy(file_path, dest_path)  # Copy the file
            print(f"Copied {file_path} to {dest_path}")
        else:
            print(f"{file_path} is not a valid file.")