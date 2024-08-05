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

def optical_flow_accumulate(file_path):
    # Open the video file
    cap = cv2.VideoCapture(file_path)

    # Read the first frame
    ret, previous_frame = cap.read()
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # Create an array to accumulate movement
    movement_accumulator = np.zeros_like(previous_frame_gray, dtype=np.float32)

    count = 0
    while cap.isOpened():
        count += 1
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Compute the dense optical flow
        flow = cv2.calcOpticalFlowFarneback(previous_frame_gray, current_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute the magnitude of the flow vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Accumulate the magnitude of the flow
        movement_accumulator += magnitude
        # Update the previous frame
        previous_frame_gray = current_frame_gray
        if count == 4400:
            break

    # Normalize the accumulator for visualization
    movement_accumulator = cv2.normalize(movement_accumulator, None, 0, 255, cv2.NORM_MINMAX)
    movement_accumulator = movement_accumulator.astype(np.uint8)

    # Display the accumulated movement
    cv2.imshow('Movement Accumulator', movement_accumulator)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

def video_window(video_path, save_path, width, height, window_dim):
    # Define the path to the input and output video
    input_video_path = video_path
    output_video_path = save_path

    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get the properties of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the window coordinates (top-left corner x, y) and dimensions (width, height)
    x, y, w, h = width, height, window_dim[0], window_dim[1]  # Example coordinates and dimensions

    # Ensure the window dimensions are within the frame size
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

        # Extract the window from the frame
        window = frame[y:y+h, x:x+w]

        # Write the window to the output video
        out.write(window)

        frame_count += 1

    # Release the video capture and writer objects
    cap.release()
    out.release()

    print(f"Extracted {frame_count} frames and saved to '{output_video_path}'")