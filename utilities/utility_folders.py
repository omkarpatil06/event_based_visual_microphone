import os
import re
import cv2
import numpy as np
import soundfile as sf
import sounddevice as sd

def get_sound(file_path):
    return sf.read(file_path)

def play_sound(audio, sampling_rate):
    sd.play(audio, sampling_rate)
    sd.wait()

def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else float('inf')

def get_files(folder_path):
    files_list = None
    for _, _, files in os.walk(folder_path):
        files_list = [f'{folder_path}{file}' for file in files if not file.startswith('.')]
        break 
    files_list = sorted(files_list, key=lambda x: extract_number(os.path.basename(x)))
    return files_list

def get_files_name(folder_path):
    files_list = None
    for _, _, files in os.walk(folder_path):
        files_list = [f'{file}' for file in files if not file.startswith('.')]
        break 
    files_list = sorted(files_list, key=lambda x: extract_number(os.path.basename(x)))
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

def convert_avi_to_mp4(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.avi'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_name = os.path.splitext(file_name)[0] + '.mp4'
            output_file_path = os.path.join(output_folder, output_file_name)
            # Open the .avi file
            cap = cv2.VideoCapture(input_file_path)
            # Get video properties
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))
            # Convert the video frame by frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            # Release resources
            cap.release()
            out.release()
            print(f"Converted: {file_name} to {output_file_path}")