import cv2
import math
import random
import numpy as np
from scipy.stats import entropy
from scipy.signal import correlate
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt

def find_roi(heatmap, title):
    region_size = (30, 30)
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

def video_moments(video_path):
    def update(frame, frame_count, mean_prev, moment2_prev, moment3_prev, moment4_prev):
        frame_count += 1
        alpha_n = 1.0/frame_count
        beta_n = frame - mean_prev
        mean = (1.0 - alpha_n) * mean_prev + alpha_n * frame
        gamma_n = frame - mean
        moment2 = moment2_prev + beta_n * gamma_n
        moment3 = moment3_prev + (1.0 - alpha_n) * beta_n * (gamma_n**2) - alpha_n * beta_n * (gamma_n**2)
        moment4 = moment4_prev + ((1.0 - alpha_n)**2) * beta_n * (gamma_n**3) - 3 * alpha_n * (1.0 - alpha_n) * beta_n * (gamma_n**3) + 3 * (alpha_n**2) * beta_n * (gamma_n**3) - 3 * moment2 * beta_n
        return frame_count, mean, moment2, moment3, moment4

    # create cv2 video read object and read first frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None
    gray_frame = np.array(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), dtype=np.float32)

    # initialise process parameters
    frame_count = 0
    mean_prev = np.zeros_like(gray_frame)
    moment2_prev = np.zeros_like(gray_frame)
    moment3_prev = np.zeros_like(gray_frame)
    moment4_prev = np.zeros_like(gray_frame)

    # calculate stats for first frame
    frame_count, mean_prev, moment2_prev, moment3_prev, moment4_prev = update(gray_frame, frame_count, mean_prev, moment2_prev, moment3_prev, moment4_prev)

    # calculate for rest of the frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count, mean_prev, moment2_prev, moment3_prev, moment4_prev = update(gray_frame, frame_count, mean_prev, moment2_prev, moment3_prev, moment4_prev)
        if frame_count % 1000 == 0:
            print(f"Total frames processed for moments: {frame_count}")
    cap.release()

    # calculate the final mean, variance, skewness and kurtosis
    mean = mean_prev
    variance = moment2_prev / frame_count
    skewness = (np.sqrt(frame_count) * moment3_prev) / (np.sqrt(moment2_prev))**3
    kurtosis = (frame_count * moment4_prev) / (moment2_prev**2) - 3

    mean_roi = find_roi(mean, "Video Mean")
    variance_roi = find_roi(variance, "Video Variance")
    skewness_roi = find_roi(skewness, "Video Skewness")
    kurtosis_roi = find_roi(kurtosis, "Video Kurtosis")
    return mean_roi, variance_roi, skewness_roi, kurtosis_roi

def video_temporal_motion(video_path):
    # create cv2 video read object and read first frame
    frame_count = 1
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None
    gray_frame_prev = np.array(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), dtype=np.float32)
    motion_sum = np.zeros_like(gray_frame_prev, dtype=np.float32)

    # calculate motion for rest of the frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_frame = cv2.calcOpticalFlowFarneback(gray_frame_prev, gray_frame, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        magnitude_frame, _ = cv2.cartToPolar(flow_frame[..., 0], flow_frame[..., 1])
        # update for next step
        frame_count += 1
        gray_frame_prev = gray_frame
        motion_sum += magnitude_frame
        if frame_count % 1000 == 0:
            print(f"Total frames processed for motion frame: {frame_count}")
    cap.release()
    
    # calculate average motion frame
    average_motion = motion_sum/frame_count
    avg_motion_roi = find_roi(average_motion, "Video Average Motion Frame")
    return avg_motion_roi

def video_frame_differencing(video_path):
    # create cv2 video read object and read first frame
    frame_count = 0
    threshold = 25
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
        # update for next step
        frame_count += 1
        gray_frame_prev = gray_frame
        motion_energy_map += thresh_frame.astype(np.float32)
        if frame_count % 1000 == 0:
            print(f"Total frames processed for average difference frame: {frame_count}")
    cap.release()
    motion_energy_normalised = cv2.normalize(motion_energy_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    motion_energy_roi = find_roi(motion_energy_normalised, "Video Difference Frame")
    return motion_energy_roi

def video_phase_variance(video_path):
    # implementation of a raised cosine filter 
    def rcosFn(width, position, values):
        sz = 256
        X = np.pi * np.arange(-sz-1, 2, 1) / (2*sz)
        Y = values[0] + (values[1]-values[0]) * np.cos(X)**2
        Y[0] = Y[1]
        Y[sz+2] = Y[sz+1]
        X = position + (2*width/np.pi) * (X + np.pi/4)
        return X, Y

    def buildRaisedCosineFilter(shape, width=1):
        rows, cols = shape
        Xrcos, Yrcos = rcosFn(width, (-width/2), [0, 1])
        Yrcos = np.sqrt(Yrcos)  # Get the square root for band-pass filtering
        
        # Create 2D grids of frequencies
        row_freqs = np.fft.fftfreq(rows)
        col_freqs = np.fft.fftfreq(cols)
        fx, fy = np.meshgrid(col_freqs, row_freqs)
        
        log_rad = np.sqrt(fx**2 + fy**2)
        log_rad[log_rad == 0] = 1e-10  # Avoid division by zero
        log_rad = np.log2(log_rad)
        
        YIrcos = np.sqrt(1.0 - Yrcos**2)
        lo_mask = np.interp(log_rad.flatten(), Xrcos, YIrcos).reshape(rows, cols)
        
        return lo_mask

    def applyRaisedCosineFilter(fft_image, lo_mask):
        return fft_image * lo_mask

    # create cv2 video read object and read first frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None
    gray_frame_prev = np.array(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), dtype=np.float32)
    
    # initialising parameters for the process
    frame_count = 0
    rows, cols = gray_frame_prev.shape
    lo_mask = buildRaisedCosineFilter((rows, cols), width=1)
    mean_phase = np.zeros_like(gray_frame_prev, dtype=np.float64)
    M2_phase = np.zeros_like(gray_frame_prev, dtype=np.float64)

    # calculate motion frames phase response
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # calculate fft
        gray_frame = np.array(gray_frame)
        fft_image = fftshift(fft2(gray_frame))
        filtered_fft = applyRaisedCosineFilter(fft_image, lo_mask)
        filtered_image = np.real(ifft2(ifftshift(filtered_fft)))
        phase_response = np.angle(filtered_image)

        # update for next step
        frame_count += 1
        delta = phase_response - mean_phase
        mean_phase += delta / frame_count
        delta2 = phase_response - mean_phase
        M2_phase += delta * delta2
        gray_frame_prev = gray_frame
        if frame_count % 100 == 0:
            print(f"Total frames processed for phase variance: {frame_count}")
    cap.release()

    # calulating variance of phase frames
    variance_phase = M2_phase / frame_count
    phase_variance_normalized = cv2.normalize(variance_phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    phase_variance_roi = find_roi(phase_variance_normalized, "Motion Video's Phase Response Variance")
    return phase_variance_roi


def video_motion_phase_variance(video_path):
    # implementation of a raised cosine filter 
    def rcosFn(width, position, values):
        sz = 256
        X = np.pi * np.arange(-sz-1, 2, 1) / (2*sz)
        Y = values[0] + (values[1]-values[0]) * np.cos(X)**2
        Y[0] = Y[1]
        Y[sz+2] = Y[sz+1]
        X = position + (2*width/np.pi) * (X + np.pi/4)
        return X, Y

    def buildRaisedCosineFilter(shape, width=1):
        rows, cols = shape
        Xrcos, Yrcos = rcosFn(width, (-width/2), [0, 1])
        Yrcos = np.sqrt(Yrcos)  # Get the square root for band-pass filtering
        
        # Create 2D grids of frequencies
        row_freqs = np.fft.fftfreq(rows)
        col_freqs = np.fft.fftfreq(cols)
        fx, fy = np.meshgrid(col_freqs, row_freqs)
        
        log_rad = np.sqrt(fx**2 + fy**2)
        log_rad[log_rad == 0] = 1e-10  # Avoid division by zero
        log_rad = np.log2(log_rad)
        
        YIrcos = np.sqrt(1.0 - Yrcos**2)
        lo_mask = np.interp(log_rad.flatten(), Xrcos, YIrcos).reshape(rows, cols)
        
        return lo_mask

    def applyRaisedCosineFilter(fft_image, lo_mask):
        return fft_image * lo_mask

    # create cv2 video read object and read first frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None
    gray_frame_prev = np.array(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), dtype=np.float32)
    
    # initialising parameters for the process
    frame_count = 0
    flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    rows, cols = gray_frame_prev.shape
    lo_mask = buildRaisedCosineFilter((rows, cols), width=1)
    mean_phase = np.zeros_like(gray_frame_prev, dtype=np.float64)
    M2_phase = np.zeros_like(gray_frame_prev, dtype=np.float64)

    # calculate motion frames phase response
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        flow_frame = cv2.calcOpticalFlowFarneback(gray_frame_prev, gray_frame, None, **flow_params)
        magnitude_frame, _ = cv2.cartToPolar(flow_frame[..., 0], flow_frame[..., 1])
        
        # calculate fft
        motion_frame = np.array(magnitude_frame)
        fft_image = fftshift(fft2(motion_frame))
        filtered_fft = applyRaisedCosineFilter(fft_image, lo_mask)
        filtered_image = np.real(ifft2(ifftshift(filtered_fft)))
        phase_response = np.angle(filtered_image)

        # update for next step
        frame_count += 1
        delta = phase_response - mean_phase
        mean_phase += delta / frame_count
        delta2 = phase_response - mean_phase
        M2_phase += delta * delta2
        gray_frame_prev = gray_frame
        if frame_count % 100 == 0:
            print(f"Total frames processed for motion phase variance: {frame_count}")
    cap.release()

    # calulating variance of phase frames
    variance_phase = M2_phase / frame_count
    phase_variance_normalized = cv2.normalize(variance_phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    phase_variance_roi = find_roi(phase_variance_normalized, "Motion Video's Phase Response Variance")
    return phase_variance_roi

def normalise_signal(signal):
    signal = signal - np.mean(signal)
    return signal / np.std(signal)

def cross_correlate_signals(video_path, reference_signal, iterations=500):
    def extract_pixel_value(video_path, x, y):
        # create cv2 video read object and read first frame
        pixel_values = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        while True:
            ret, frame = cap.read()
            if not ret:
                break    
            # Convert to grayscale and append pixel value
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pixel_values.append(gray_frame[y, x])
        cap.release()
        return np.array(pixel_values)

    def correlate_pixel(video_path, reference_signal, x, y):
        pixel_values = extract_pixel_value(video_path, x, y)
        extracted_signal = normalise_signal(pixel_values)
        correlation = correlate(extracted_signal, reference_signal, mode='full')
        max_corr = np.max(correlation)
        return max_corr

    # start process of finding maximum correlation
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_corr_matrix =  np.zeros((height, width))

    # start random search
    current_y = random.randint(0, height - 1)
    current_x = random.randint(0, width - 1)
    current_corr = correlate_pixel(video_path, reference_signal, current_x, current_y)
    
    best_y, best_x = current_y, current_x
    best_corr = current_corr
    
    temp = 10.0
    cooling_rate = 0.99
    
    for i in range(iterations):
        # Generate a random neighboring pixel
        neighbor_y = min(max(current_y + random.randint(-30, 30), 0), height - 1)
        neighbor_x = min(max(current_x + random.randint(-30, 30), 0), width - 1)
        
        neighbor_corr = correlate_pixel(video_path, reference_signal, neighbor_x, neighbor_y)
        
        # Decide whether to move to the neighbor
        if neighbor_corr > current_corr:
            current_y, current_x = neighbor_y, neighbor_x
            current_corr = neighbor_corr
            max_corr_matrix[current_y, current_x] = current_corr
        else:
            # Accept the move with a probability depending on the temperature
            delta_corr = neighbor_corr - current_corr
            acceptance_probability = math.exp(delta_corr / temp)
            if random.random() < acceptance_probability:
                current_y, current_x = neighbor_y, neighbor_x
                current_corr = neighbor_corr
                max_corr_matrix[current_y, current_x] = current_corr
        
        # Update the best found position
        if current_corr > best_corr:
            best_y, best_x = current_y, current_x
            best_corr = current_corr
        
        # Cool down the temperature
        temp *= cooling_rate
        print(f"Iteration {i + 1}/{iterations}: Best Correlation = {best_corr:.4f} at ({best_y}, {best_x})")    
    
    # normalising maximum correlation matrix
    max_corr_normalised = cv2.normalize(max_corr_matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    max_corr_roi = find_roi(max_corr_normalised, "Maximum Correlation's Frame")
    return max_corr_roi

def detect_edges_in_video(video_path):
    def calculate_edge_stability(edges_over_time):
        stability_map = np.zeros_like(edges_over_time[0])
        for i in range(1, len(edges_over_time)):
            stability_map += np.abs(edges_over_time[i] - edges_over_time[i - 1])
        return stability_map
    
    # create cv2 video read object and read first frame
    frame_count = 0
    edges_over_time = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    while True:
        ret, frame = cap.read()
        if not ret:
            break    
        # Convert to grayscale and append pixel value
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, threshold1=100, threshold2=200)
        edges_over_time.append(edges)
        frame_count += 1
        if frame_count % 1000 == 0:
            print(f"Total frames processed for stability edge map: {frame_count}")
    cap.release()
    stability_map = calculate_edge_stability(edges_over_time)
    stability_map_normalised = cv2.normalize(stability_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stability_map_roi = find_roi(stability_map_normalised, "Stability Map of Edges")
    return stability_map_roi

def divide_into_windows(frame, window_size):
    windows = []
    h, w = frame.shape
    for i in range(0, h, window_size):
        for j in range(0, w, window_size):
            window = frame[i:i + window_size, j:j + window_size]
            windows.append(window)
    return windows

def calculate_entropy_over_time(windows_over_time, num_windows_y, num_windows_x):
    entropy_map = np.zeros((num_windows_y, num_windows_x))

    for window_index in range(len(windows_over_time[0])):
        pixel_values_over_time = [window[window_index].flatten() for window in windows_over_time]
        histograms = [np.histogram(values, bins=256, range=(0, 256))[0] for values in pixel_values_over_time]
        entropies = [entropy(hist) for hist in histograms]
        mean_entropy = np.mean(entropies)

        y = window_index // num_windows_x
        x = window_index % num_windows_x
        entropy_map[y, x] = mean_entropy

    return entropy_map

def find_highest_entropy_window(entropy_map, window_size):
    max_entropy_index = np.unravel_index(np.argmax(entropy_map), entropy_map.shape)
    y, x = max_entropy_index
    top_left_y = y * window_size
    top_left_x = x * window_size
    return (top_left_x, top_left_y), (window_size, window_size)

def process_video_for_entropy(video_path, window_size=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file.")

    frame_count = 0
    windows_over_time = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        windows = divide_into_windows(gray_frame, window_size)
        windows_over_time.append(windows)
        frame_count += 1
        if frame_count % 1000 == 0:
            print(f"Total frames processed for entropy map: {frame_count}")
    cap.release()

    frame_height, frame_width = gray_frame.shape
    num_windows_y = frame_height // window_size
    num_windows_x = frame_width // window_size

    entropy_map = calculate_entropy_over_time(windows_over_time, num_windows_y, num_windows_x)
    return find_highest_entropy_window(entropy_map, window_size)