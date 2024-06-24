import os
import numpy as np
import soundfile as sf
import sounddevice as sd

def get_sound(file_path):
    return sf.read(file_path)

def play_sound(audio, sampling_rate):
    sd.play(audio, sampling_rate)
    sd.wait()

def get_files(folder_path):
    files_list = None
    for _, _, files in os.walk(folder_path):
        files_list = [file for file in files if not file.startswith('.')]
        break
    return files_list

def combine_audio(folder_path, save_path):
    files_list = get_files(folder_path)
    total_audio = []
    for file in files_list:
        audio, _ = sf.read(f'{folder_path}/{file}')
        total_audio.extend(audio)
    total_audio = total_audio[:4096*(len(total_audio)//4096)]
    total_audio = np.array(total_audio)
    total_audio = total_audio.reshape((-1, 4096))
    np.save(save_path, total_audio)
