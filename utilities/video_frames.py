import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_frame(frame, title):
    plt.imshow(frame, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.show()

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

def get_video_info(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError("Could not open video file")

    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    if not ret:
        raise IOError("Could not read frame from video")

    channels = frame.shape[2] if len(frame.shape) == 3 else 1
    cap.release()
    return {
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "channels": channels
    }

def video_window(video_path, save_path, width, height, window_dim):
    # Define the path to the input and output video
    input_video_path = video_path
    output_video_path = save_path

    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the window coordinates (top-left corner x, y) and dimensions (width, height)
    x, y, w, h = width, height, window_dim[0], window_dim[1]  # Example coordinates and dimensions
    if x + w > frame_width or y + h > frame_height:
        print("Error: Window dimensions exceed frame size.")
        exit()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        window = frame[y:y+h, x:x+w]
        out.write(window)
        frame_count += 1
    cap.release()
    out.release()
    print(f"Extracted {frame_count} frames and saved to '{output_video_path}'")