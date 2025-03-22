import os
import re
import shutil
import subprocess

def convert_avi_to_mp4(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for file_name in os.listdir(source_folder):
        if file_name.endswith(".avi"):
            input_file = os.path.join(source_folder, file_name)
            output_file = os.path.join(destination_folder, os.path.splitext(file_name)[0] + ".mp4")
            command = ['ffmpeg', '-i', input_file, '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', output_file]
            try:
                subprocess.run(command, check=True)
                print(f"Successfully converted: {file_name} -> {os.path.basename(output_file)}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert: {file_name}, Error: {e}")

def rename_mp4_files_in_order(directory, prefix="video"):
    mp4_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    mp4_files.sort()
    for i, filename in enumerate(mp4_files, start=1):
        new_name = f"{prefix}_{i}.mp4"
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_name}")
    print("Renaming completed!")

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

def organize_files_into_folders(source_dir, batch_size=5):
    wav_files = [f for f in os.listdir(source_dir) if f.startswith('roi_') and f.endswith('.wav')]
    wav_files.sort(key=extract_number)
    for i in range(0, len(wav_files), batch_size):
        folder_num = (i // batch_size) + 1
        target_dir = os.path.join(source_dir, f'video_{folder_num}')
        os.makedirs(target_dir, exist_ok=True)
        for j in range(i, min(i + batch_size, len(wav_files))):
            src_file = os.path.join(source_dir, wav_files[j])
            dest_file = os.path.join(target_dir, wav_files[j])
            shutil.move(src_file, dest_file)
            print(f'Moved {wav_files[j]} to {target_dir}')

def undo_file_movement(source_dir):
    video_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and d.startswith('video_')]
    for video_dir in video_dirs:
        video_path = os.path.join(source_dir, video_dir)
        wav_files = [f for f in os.listdir(video_path) if f.endswith('.wav')]
        for wav_file in wav_files:
            src_file = os.path.join(video_path, wav_file)
            dest_file = os.path.join(source_dir, wav_file)
            shutil.move(src_file, dest_file)
            print(f'Moved {wav_file} back to {source_dir}')
        os.rmdir(video_path)
        print(f'Removed folder {video_path}')

source_dir = '/Volumes/Omkar 5T/dataset/haoqi_lowest/audio/' 
organize_files_into_folders(source_dir, batch_size=5)