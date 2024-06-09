import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

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
    image = librosa.display.specshow(data=mag_matrix, sr=sr, n_fft=n_fft, vmin=-5, vmax=5, hop_length=hop_length, y_axis='log', x_axis='time', cmap='jet', ax=ax)
    ax.set_title(f'Magnitude Spectrogram of {signal_title} Signal')
    fig.colorbar(image, ax=ax, format='%+2.f dB')
    plt.show()