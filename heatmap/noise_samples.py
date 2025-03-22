import os
import sys
import shutil
import random
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
sys.path.insert(0, '/Users/omkarpatil/Documents/GitHub/event_based_visual_microphone/utilities/')
import video_frames as vf

def create_smaller_videos():
    counter = 0
    w_point = 0
    w_limit = 704
    h_point = 0
    h_limit = 200
    window_dim = [30, 30]
    video_path = '/Volumes/Omkar 5T/dataset/video_dataset/plants.mp4'
    save_path = '/Volumes/Omkar 5T/dataset/heatmap_dataset/noise_videos/plants_'

    video_width = w_limit
    video_height = h_limit

    for h_point in range(60, h_limit, 30):
        for w_point in range(0, w_limit, 30):
            # Ensure the window doesn't exceed the video frame
            if w_point + window_dim[0] <= video_width and h_point + window_dim[1] <= video_height:
                counter += 1
                save_video_path = f'{save_path}{counter}.mp4'
                vf.video_window(video_path, save_video_path, w_point, h_point, window_dim)
            else:
                print(f"Skipping window starting at ({w_point}, {h_point}) as it exceeds video dimensions.")

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

def low_pass_filter(audio, sr, cutoff_freq):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

def create_noisy_audio(clean_audio_path, noise_audio_path, output_path, snr_db=-5, resample_rate=2200):
    # Load the clean audio and resample it
    clean_audio, sr_clean = librosa.load(clean_audio_path, sr=None)
    clean_audio = librosa.resample(clean_audio, orig_sr=sr_clean, target_sr=resample_rate)
    
    # Apply a low-pass filter to the resampled audio to remove high-frequency components
    filtered_audio = low_pass_filter(clean_audio, resample_rate, cutoff_freq=1024)  # 1000 Hz cutoff
    
    # Load the noise audio and resample it to match the clean audio
    noise_audio, sr_noise = librosa.load(noise_audio_path, sr=None)
    noise_audio = librosa.resample(noise_audio, orig_sr=sr_noise, target_sr=resample_rate)
    
    # Use the shorter length between the filtered audio and the noise audio
    min_length = min(len(filtered_audio), len(noise_audio))
    filtered_audio = filtered_audio[:min_length]
    noise_audio = noise_audio[:min_length]
    
    # Adjust the volume of the filtered audio (to make it faint)
    # Calculate signal power
    signal_power = np.mean(filtered_audio**2)
    # Calculate noise power
    noise_power = np.mean(noise_audio**2)
    # Calculate the desired noise power for given SNR
    desired_noise_power = signal_power / (10**(snr_db/10))
    # Scale noise to achieve the desired power
    scaling_factor = np.sqrt(desired_noise_power / noise_power)
    scaled_noise = noise_audio * scaling_factor
    
    # Mix the scaled noise with the filtered audio
    noisy_audio = filtered_audio + scaled_noise
    
    # Normalize the noisy audio to the range [-1, 1]
    normalized_noisy_audio = normalize_audio(noisy_audio)
    
    # Save the noisy audio
    sf.write(output_path, normalized_noisy_audio, resample_rate)
    
    print(f"Noisy audio saved to: {output_path}")

def generate_noisy_samples(audio_folder, noise_folder, output_folder, num_samples=512, num_pure_samples=50, snr_db=-10):
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')]
    noise_files = [os.path.join(noise_folder, f) for f in os.listdir(noise_folder) if f.endswith('.wav')]
    total_samples = num_samples + num_pure_samples
    
    for i in range(total_samples):
        output_file = os.path.join(output_folder, f'noisy_sample_{i+1}.wav')
        
        if i < num_pure_samples:
            # Copy pure audio files
            pure_audio_file = random.choice(audio_files)
            shutil.copy(pure_audio_file, output_file)
            print(f"Copied pure audio file to {output_file}")
        else:
            # Randomly select an audio file and a noise file
            audio_file = random.choice(audio_files)
            noise_file = random.choice(noise_files)
            # Create and save the noisy audio sample
            create_noisy_audio(audio_file, noise_file, output_file, snr_db=snr_db)
            print(f"Created noisy audio file {output_file}")
