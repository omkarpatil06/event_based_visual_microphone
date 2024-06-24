import sys
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

sys.path.insert(0, '/Users/omkarpatil/Documents/event_based_visual_microphone/utilities/')
import utility_spectrogram as us
import utility_folders as uf

def normalise_signal(signal):
    maxsx = np.max(signal)
    minsx = np.min(signal)
    if maxsx!=1.0 or minsx!=-1.0:
        range_val = maxsx - minsx
        signal = 2*signal/range_val
        newmx = np.max(signal)
        offset = newmx-1.0
        signal = signal-offset
    return signal

def lowpass_filter(audio, orignal_freq, downsampling_freq):
    nyquist_freq = 0.5*orignal_freq
    normal_cutoff = downsampling_freq/nyquist_freq
    b, a = butter(9, normal_cutoff, btype='low', analog=False)
    audio = filtfilt(b, a, audio)
    return audio

def right_pad(audio, total_samples):
    if len(audio) >= total_samples:
        audio = audio[:total_samples]
    else:
        audio = np.array(audio)
        pad_length = total_samples - len(audio)
        audio = np.pad(audio, (0, pad_length), 'constant', constant_values=(0))
    return audio

def aliased_dsample(audio, downsampling_factor):
    audio = audio[::downsampling_factor]
    return audio

def nonaliased_dsample(audio, orignal_freq, downsampling_freq):
    audio = lowpass_filter(audio, orignal_freq, downsampling_freq)
    audio = librosa.resample(audio, orig_sr=orignal_freq, target_sr=downsampling_freq)
    return audio

def upsample(audio, upsampling_freq, orignal_freq):
    audio = librosa.resample(audio, orig_sr=orignal_freq, target_sr=upsampling_freq)
    return audio

def aliased_dsample_folder(folder_path, save_path, duration, downsampling_factor, orignal_freq):
    files_list = uf.get_files(folder_path)
    for file in files_list:
        audio, _ = sf.read(f'{folder_path}/{file}')
        audio = normalise_signal(audio)
        audio = aliased_dsample(audio, downsampling_factor)

        downsampled_freq = orignal_freq/downsampling_factor
        audio = upsample(audio, orignal_freq, downsampled_freq)
        total_samples = int(duration*orignal_freq)
        audio = right_pad(audio, total_samples)
        audio = normalise_signal(audio)
        sf.write(f'{save_path}/{file}', audio, int(orignal_freq))

def nonaliased_dsample_folder(folder_path, save_path, duration, downsampling_freq, orignal_freq):
    files_list = uf.get_files(folder_path)
    for file in files_list:
        audio, _ = sf.read(f'{folder_path}/{file}')
        audio = normalise_signal(audio)
        audio = nonaliased_dsample(audio, orignal_freq, downsampling_freq)

        total_samples = int(duration*downsampling_freq)
        audio = right_pad(audio, total_samples)
        audio = normalise_signal(audio)

        sf.write(f'{save_path}/{file}', audio, downsampling_freq)