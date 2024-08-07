import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

# UTILITY SIGNALS:
def show_signal(signal, signal_title):
    fig, ax = plt.subplots(figsize=(8, 5))
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
    fig, ax = plt.subplots(figsize=(8, 5))
    image = librosa.display.specshow(data=mag_matrix, sr=sr, n_fft=n_fft, vmin=-6.5, vmax=6.5, hop_length=hop_length, y_axis='log', x_axis='time', cmap='jet', ax=ax)
    ax.set_title(f'Magnitude Spectrogram of {signal_title} Signal')
    fig.colorbar(image, ax=ax, format='%+2.f dB')
    plt.show()

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