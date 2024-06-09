import os
import cv2
import numpy as np

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames.")
    return np.array(frames)

def extract_video_segments(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size_bytes = frame_width * frame_height * 3
    frames_per_segment = np.floor(2*(1024**3)/frame_size_bytes).astype(np.int32)
    total_segments = np.ceil(total_frames/frames_per_segment).astype(np.int32)

    desktop_path = os.path.join(os.path.expanduser("~"), "Documents")
    new_folder_path = os.path.join(desktop_path, 'TemporaryVideoSegments')
    try:
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Temporary folder created: {new_folder_path}")
    except Exception as e:
        print(f"Error creating temporary folder: {e}")

    interval = 0
    segment_count = 0
    while total_frames > interval:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, interval)
        frames = []

        for frame_idx in range(interval, interval+frames_per_segment-1):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        segment_count += 1
        file_path = os.path.join(new_folder_path, f'Segment{segment_count}.npy')
        frames = np.array(frames)
        np.save(file_path, frames)
        print(f"Saved video segment {segment_count}/{total_segments} to Documents.")
        interval += frames_per_segment