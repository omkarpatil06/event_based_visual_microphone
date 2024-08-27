import cv2
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappushpop

#################################################################
# code for pre-processing video
#################################################################
def gaussian_blurred_video(video_path, save_path, sigma_base):
    # FUNCTIONS FOR APPLYING GAUSSIAN BLUR
    def apply_amplitude_weighted_blur(response, sigma_base):
        amplitude = np.abs(response)
        phase = np.angle(response)
        normalized_amplitude = np.maximum(amplitude / np.max(amplitude), 1e-10)  # Avoid division by zero

        sigma_levels = {'high': sigma_base * 0.5, 
                        'medium': sigma_base * 0.6, 
                        'low': sigma_base * 0.7, 
                        'very_low': sigma_base * 0.8}
        blurred_regions = {}
        for key, sigma in sigma_levels.items():
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size <= 0:
                kernel_size = 1  
            blurred_regions[key] = cv2.GaussianBlur(phase, (kernel_size, kernel_size), sigma)
        blurred_phase = np.zeros_like(phase)
        for key, mask in [('high', normalized_amplitude > 0.75),
                          ('medium', (normalized_amplitude > 0.5) & (normalized_amplitude <= 0.75)),
                          ('low', (normalized_amplitude > 0.25) & (normalized_amplitude <= 0.5)),
                          ('very_low', normalized_amplitude <= 0.25)]:
            blurred_phase = np.where(mask, blurred_regions[key], blurred_phase)
        return amplitude * np.exp(1j * blurred_phase)

    def preprocess_step(frame, sigma_base):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)  
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        blurred_f_shift = apply_amplitude_weighted_blur(f_shift, sigma_base)
        f_ishift = np.fft.ifftshift(blurred_f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img_back

    # create cv2 video read object
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return
    
    # cv2 process parameters
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # create cv2 video write object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap_out = cv2.VideoWriter(save_path, fourcc, fps, (width, height), isColor=False)

    # code for processing the video with gaussian blur
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = preprocess_step(frame, sigma_base)
        cap_out.write(processed_frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Total frames processed with pre-process step : {frame_count}")
    cap.release()
    cap_out.release()
    print(f"Pre-processed {frame_count} frames and saved to '{save_path}'")

#################################################################
# code for finding ROI's
#################################################################
def find_roi(heatmap, title, region_size = (30, 30)):
    matrix = np.array(heatmap)
    rows, cols = matrix.shape
    region_rows, region_cols = region_size

    # initialising before search
    max_sum = -np.inf
    top_left_coordinates = (0, 0)

    # start search
    for i in range(rows - region_rows + 1):
        for j in range(cols - region_cols + 1):
            current_sum = np.sum(matrix[i : i + region_rows, j : j + region_cols])
            if current_sum > max_sum:
                max_sum = current_sum
                top_left_coordinates = (i, j)
    
    # display the results
    plt.imshow(matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    rect = plt.Rectangle((top_left_coordinates[1] - 0.5, top_left_coordinates[0] - 0.5), region_size[1], region_size[0], edgecolor='red', facecolor='none', linewidth=2)
    plt.gca().add_patch(rect)
    plt.title(f"Brightest Region for {title}")
    plt.show()
    return top_left_coordinates


def find_top_n_rois(heatmap, title, region_size=(30, 30), top_n=3, min_distance=30):
    def is_too_close(new_roi, existing_rois, min_dist):
        new_y, new_x = new_roi
        for _, (ey, ex) in existing_rois:
            if np.sqrt((new_y - ey)**2 + (new_x - ex)**2) < min_dist:
                return True
        return False

    matrix = np.array(heatmap)
    rows, cols = matrix.shape
    region_rows, region_cols = region_size

    # initialising before search
    regions = []

    # start search
    for i in range(rows - region_rows + 1):
        for j in range(cols - region_cols + 1):
            current_sum = np.sum(matrix[i: i + region_rows, j: j + region_cols])
            regions.append((current_sum, (i, j)))
    regions.sort(reverse=True, key=lambda x: x[0])
    
    selected_rois = []
    for current_sum, (i, j) in regions:
        if len(selected_rois) < top_n:
            if not is_too_close((i, j), selected_rois, min_distance):
                selected_rois.append((current_sum, (i, j)))
            if len(selected_rois) == top_n:
                break

    # display results
    plt.imshow(matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    for score, (i, j) in selected_rois:
        rect = plt.Rectangle((j, i), region_cols, region_rows, edgecolor='red', facecolor='none', linewidth=2)
        plt.gca().add_patch(rect)
    plt.title(f"Top {top_n} Brightest and Separated Regions for {title}")
    plt.show()
    return [(i, j) for _, (i, j) in selected_rois]

#################################################################
# code for difference frame calculations
#################################################################
def video_threshold(video_path, tail_percentage):
    # create cv2 video read object and read first frame
    frame_count = 0
    threshold = 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None
    gray_frame_prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

     # calculate motion for rest of the frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        difference_frame = cv2.absdiff(gray_frame_prev, gray_frame)

        hist, bins = np.histogram(difference_frame.ravel(), bins=256, range=(0, 256))
        cumulative_hist = np.cumsum(hist)
        total_pixels = cumulative_hist[-1]
        threshold_index = np.where(cumulative_hist >= (1 - tail_percentage / 100) * total_pixels)[0][0]
        threshold += bins[threshold_index]
        frame_count += 1
        gray_frame_prev = gray_frame
        if frame_count % 1000 == 0:
            print(f"Total frames processed for threshold: {frame_count}")
    cap.release()
    threshold = threshold/frame_count
    print(f"The video threshold is: {threshold}.")
    return threshold

def video_avg_difference_frame(video_path, threshold):
    # create cv2 video read object and read first frame
    frame_count = 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None
    gray_frame_prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    motion_energy_map = np.zeros_like(gray_frame_prev, dtype=np.float32)

    # calculate motion for rest of the frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        difference_frame = cv2.absdiff(gray_frame_prev, gray_frame)
        _, thresh_frame = cv2.threshold(difference_frame, threshold, 255, cv2.THRESH_BINARY)
        motion_energy_map += thresh_frame.astype(np.float32)
        # update for next step
        frame_count += 1
        gray_frame_prev = gray_frame
        if frame_count % 1000 == 0:
            print(f"Total frames processed for average difference frame: {frame_count}")
    cap.release()
    motion_energy_normalised = cv2.normalize(motion_energy_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return motion_energy_normalised

def calculate_zero_crossings(video_path):
    # create cv2 video read object
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the first frame")
        cap.release()
        return None
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_diff = np.zeros_like(prev_frame_gray, dtype=np.float32)
    zero_crossings = np.zeros_like(prev_frame_gray, dtype=np.int32)
    
    # start the zero-crossing video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_diff = np.float32(current_frame_gray) - np.float32(prev_frame_gray)
        zero_crossings += ((prev_diff > 0) & (current_diff < 0)) | ((prev_diff < 0) & (current_diff > 0))
        prev_frame_gray = current_frame_gray
        prev_diff = current_diff
    cap.release()
    zero_crossings_normalised = cv2.normalize(zero_crossings, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return zero_crossings_normalised

#################################################################
# save video based on ROI 
#################################################################
def find_top_pixels_within_roi(heatmap, roi_top_left, roi_size, top_n):
    x, y = roi_top_left
    roi_width, roi_height = roi_size
    roi = heatmap[x:x + roi_height, y:y + roi_width]
    flat_indices = np.argsort(-roi.ravel())[:top_n]
    brightest_coords = np.unravel_index(flat_indices, roi.shape)
    return [(x + dx, y + dy) for dx, dy in zip(brightest_coords[0], brightest_coords[1])]

def create_highlighted_video(heatmap, video_path, save_path, rois, roi_size, neighborhood_size=2, top_n=10):
    # create cv2 video read and write object
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))), True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = np.zeros_like(frame, dtype=np.uint8)
        for roi_top_left in rois:
            top_pixels = find_top_pixels_within_roi(heatmap, roi_top_left, roi_size, top_n)
            for (px, py) in top_pixels:
                x_min = max(0, px - neighborhood_size)
                x_max = min(frame.shape[0], px + neighborhood_size + 1)
                y_min = max(0, py - neighborhood_size)
                y_max = min(frame.shape[1], py + neighborhood_size + 1)
                mask[x_min:x_max, y_min:y_max] = frame[x_min:x_max, y_min:y_max]
        out.write(mask)
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to '{save_path}'")