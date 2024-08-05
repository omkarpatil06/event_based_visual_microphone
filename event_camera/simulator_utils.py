import os
import re
import cv2
import torch
import numpy as np
import torch.nn.functional as F

#################################################################
# RGB TO EVENT VIDEO UTILITY FUNCTIONS
#################################################################
def lin_log(input_frame):
    thrshold = 20
    rounding = 1e8
    input_frame = input_frame.double()
    freq = (1./thrshold)*np.log(thrshold)
    linlog_frame = torch.where(input_frame <= thrshold, input_frame*freq, torch.log(input_frame))
    linlog_frame = torch.round(linlog_frame*rounding)/rounding
    return linlog_frame.float()

def rescale_intensity_frame(input_frame):
    return (input_frame+20)/275.

def low_pass_filter(cutoff_freq, new_frame, mem_frame, intensity_frame, delta_time):
    tau = 1/(2*np.pi*cutoff_freq)
    epsilon = (delta_time/tau)*intensity_frame
    epsilon = torch.clamp(epsilon, max=1)
    new_frame = (1-epsilon)*mem_frame + epsilon*new_frame
    return new_frame

def compute_event_map(diff_frame, pos_thresh, neg_thresh):
    pos_frame = F.relu(diff_frame)
    neg_frame = F.relu(-diff_frame)
    # print(f'Min difference any pixel: {min(diff_frame.min(), diff_frame.min())}')
    # print(f'Max difference any pixel: {max(diff_frame.max(), diff_frame.max())}')
    pos_evts_frame = torch.div(pos_frame, pos_thresh, rounding_mode='floor').type(torch.int32)
    neg_evts_frame = torch.div(neg_frame, neg_thresh, rounding_mode='floor').type(torch.int32)
    print(f'HELLO!: Max events any pixel: {max(pos_evts_frame.max(), neg_evts_frame.max())}')
    return pos_evts_frame, neg_evts_frame

#################################################################
# RGB TO EVENT VIDEO CONVERSION UTILITY FUNCTIONS
#################################################################
def generate_event_frames(input_frame, mem_frame, delta_time, cutoff_freq, pos_thresh, neg_thresh):
    max_intensity_events = 16
    # RGB to Luma
    input_frame = input_frame.astype(np.uint8)
    luma_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    luma_frame = luma_frame.astype(np.float32)
    luma_frame = torch.tensor(luma_frame, dtype=torch.float64)
    # Luma to lin-log
    linlog_frame = lin_log(luma_frame)
    # Bandpass pixel filter
    intensity_frame = rescale_intensity_frame(luma_frame)
    if mem_frame is None:
        mem_frame = linlog_frame
    lp_frame = low_pass_filter(cutoff_freq, linlog_frame, mem_frame, intensity_frame, delta_time)
    # Generate event frames
    diff_frame =lp_frame-mem_frame
    pos_evts_frame, neg_evts_frame = compute_event_map(diff_frame, pos_thresh, neg_thresh)
    pos_evts_frame = torch.clamp(pos_evts_frame, max=max_intensity_events).type(torch.int32)
    neg_evts_frame = torch.clamp(neg_evts_frame, max=max_intensity_events).type(torch.int32)
    return pos_evts_frame, neg_evts_frame, lp_frame

def event_video_frames(pos_evts_frame, neg_evts_frame):
    max_intensity_events = 16
    evts_frame = pos_evts_frame - neg_evts_frame
    evts_frame = 255*(evts_frame + max_intensity_events)/(2*max_intensity_events)
    evts_frame = torch.clamp(evts_frame, max=255).type(torch.uint8)
    return evts_frame

#################################################################
# RGB TO EVENT VIDEO CONVERSION FUNCTION
#################################################################
def event_video(rgb_frames, delta_time, cutoff_freq, pos_thresh, neg_thresh):
    mem_frame = None
    reconstructed_frames = []
    pos_frames, neg_frames = [], []
    for frame in rgb_frames:
        pos_evts_frame, neg_evts_frame, mem_frame = generate_event_frames(frame, mem_frame, delta_time, cutoff_freq, pos_thresh, neg_thresh) 
        evts_frame = event_video_frames(pos_evts_frame, neg_evts_frame)
        pos_frames.append(pos_evts_frame)
        neg_frames.append(neg_evts_frame)
        reconstructed_frames.append(evts_frame)
    return np.array(pos_frames), np.array(neg_frames), np.array(reconstructed_frames)

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def event_simulator(delta_time, cutoff_freq, pos_thresh, neg_thresh):
    desktop_path = os.path.join(os.path.expanduser("~"), "Documents")
    folder_path = os.path.join(desktop_path, 'TemporaryVideoSegments')
    video_files = None
    for _, _, files in os.walk(folder_path):
        video_files = [file for file in files if not file.startswith('.')]
        break
    video_files.sort(key=extract_number)
    num_files = len(video_files)

    new_folder_path = os.path.join(desktop_path, 'TemporaryEventVideoSegments')
    try:
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Temporary folder created: {new_folder_path}")
    except Exception as e:
        print(f"Error creating temporary folder: {e}")
    
    segment_count = 0
    for file in video_files:
        segment_count += 1
        file_path = os.path.join(folder_path, file)
        save_path = os.path.join(new_folder_path, file)
        frames = np.load(file_path)
        _, _, evts_video = event_video(frames, delta_time, cutoff_freq, pos_thresh, neg_thresh)
        np.save(save_path, evts_video)
        print(f"Converted and saved video segment {segment_count}/{num_files} to Documents.")
        os.remove(file_path)
    os.rmdir(folder_path)

def show_video():
    desktop_path = os.path.join(os.path.expanduser("~"), "Documents")
    new_folder_path = os.path.join(desktop_path, 'TemporaryEventVideoSegments')
    file_path = os.path.join(new_folder_path, 'Segment5.npy')

    frames = np.load(file_path)
    total_frames = frames.shape[0]
    show_frames = np.floor(total_frames/7).astype(np.int32)
    frames = frames[0:show_frames]

    window_name = 'Video Playback'
    frame_rate = 30
    delay = 1/frame_rate
    cv2.namedWindow(window_name)
    for frame in frames:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(int(delay*1000)) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def save_video_from_tensor(tensor, output_path, fps=90):
    num_frames, height, width = tensor.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)
    for i in range(num_frames):
        frame = tensor[i]
        out.write(frame)
    out.release()

def save_video(save_vid, video_save_path, fps):
    if save_vid:
        desktop_path = os.path.join(os.path.expanduser("~"), "Documents")
        folder_path = os.path.join(desktop_path, 'TemporaryEventVideoSegments')
        video_files = None
        for _, _, files in os.walk(folder_path):
            video_files = [file for file in files if not file.startswith('.')]
            break
        video_files.sort(key=extract_number)
        num_files = len(video_files)

        segment_count = 0
        video = None
        for file in video_files:
            segment_count += 1
            file_path = os.path.join(folder_path, file)
            frames = np.load(file_path)
            if segment_count == 1:
                print(frames.shape)
                video = frames
            else:
                video = np.vstack((video, frames))
            print(f"Converted and saved segment {segment_count}/{num_files} to Documents.")
        save_video_from_tensor(video, video_save_path, fps)
        

#################################################################
# EVENT FRAMES TO LIST CONVERSION UTILITY FUNCTIONS
#################################################################
def get_event_list(pos_evts_xy, neg_evts_xy, time):
    num_pos_evts = pos_evts_xy[0].shape[0]
    num_neg_evts = neg_evts_xy[0].shape[0]
    num_evts = num_pos_evts + num_neg_evts
    # Generate current event list
    current_event = None
    if num_evts > 0:
        current_event = torch.ones((num_evts, 4), dtype=torch.float32)
        current_event[:, 0] *= time
        # The x address (columns)
        current_event[:num_pos_evts, 1] = pos_evts_xy[1]
        current_event[num_pos_evts:, 1] = neg_evts_xy[1]
        # The y address (rows)
        current_event[:num_pos_evts, 2] = pos_evts_xy[0]
        current_event[num_pos_evts:, 2] = neg_evts_xy[0]
        current_event[num_pos_evts:, 3] *= -1
    # random_idx = torch.randperm(current_event.shape[0])
    # current_event = current_event[random_idx].view(current_event.size())
    return current_event

def event_frame_list(pos_evts_frame, neg_evts_frame, start_time, end_time, delta_time):
    max_num_events_any_pixel = max(pos_evts_frame.max(), neg_evts_frame.max())
    minimum_time_steps = max_num_events_any_pixel if max_num_events_any_pixel > 0 else 1
    time_step = delta_time/minimum_time_steps
    if time_step != 0:
        time = np.arange(start_time, end_time, time_step).astype(np.float32)
        events_list = torch.empty((0, 4), dtype=torch.float32)
        for event_idx in range(max_num_events_any_pixel):
            pos_cord = torch.tensor(pos_evts_frame >= event_idx + 1, dtype=torch.bool)
            neg_cord = torch.tensor(neg_evts_frame >= event_idx + 1, dtype=torch.bool)
            pos_evts_xy = pos_cord.nonzero(as_tuple=True)
            neg_evts_xy = neg_cord.nonzero(as_tuple=True)
            current_event_list = get_event_list(pos_evts_xy, neg_evts_xy, time[event_idx])
            events_list = torch.cat((events_list, current_event_list))
        return events_list

#################################################################
# EVENT FRAME TO LIST CONVERSION FUNCTION
#################################################################
def event_list(pos_frames, neg_frames, delta_time):
    start_time = 0.
    end_time = start_time + delta_time
    events_list = torch.empty((0, 4), dtype=torch.float32)
    for pos_frame, neg_frame in zip(pos_frames, neg_frames):
        current_list = event_frame_list(pos_frame, neg_frame, start_time, end_time, delta_time)
        if current_list is not None:
            events_list = torch.cat((events_list, current_list))
        start_time, end_time = end_time, end_time + delta_time
    return events_list

#################################################################
# EVENT LIST TO FRAME CONVERSION UTILITY FUNCTIONS
#################################################################
def hist2d_numba_seq(tracks, bins, ranges):
    H = np.zeros((bins[0], bins[1]), dtype=np.float64)
    delta = 1/((ranges[:, 1] - ranges[:, 0])/bins)
    for t in range(tracks.shape[1]):
        i = np.clip(int((tracks[0, t] - ranges[0, 0]) * delta[0]), 0, bins[0] - 1)
        j = np.clip(int((tracks[1, t] - ranges[1, 0]) * delta[1]), 0, bins[1] - 1)
        H[i, j] += 1
    return H

def accumulate_event_frame(events_list, histrange, height, width):
    max_intensity_events = 16
    polarity_on = (events_list[:, 3] == 1).bool().numpy().astype(bool)
    polarity_off = ~polarity_on.astype(bool)

    bins = np.asarray([height, width], dtype=np.int64)
    on_xy = np.array([events_list[polarity_on, 2], events_list[polarity_on, 1]], dtype=np.float64)
    off_xy = np.array([events_list[polarity_off, 2], events_list[polarity_off, 1]], dtype=np.float64)
    frame_on = hist2d_numba_seq(on_xy, bins, histrange)
    frame_off = hist2d_numba_seq(off_xy, bins, histrange)

    current_frame = np.zeros_like(frame_on)
    current_frame = np.clip(current_frame + (frame_on - frame_off), -max_intensity_events, max_intensity_events)
    return current_frame

#################################################################
# EVENT LIST TO FRAME CONVERSION FUNCTION
#################################################################
def events_list_frames(events_list, width, height, delta_time):
    time_list = events_list[:, 0]
    num_events = len(time_list)
    histrange = np.asarray([(0, v) for v in (height, width)], dtype=np.int64)
    this_frame_idx = 0

    def search_duration_idx(time, current_start, next_start):
        start_idx = np.searchsorted(time, current_start, side='left')
        end_idx = np.searchsorted(time, next_start, side='right')
        return start_idx, end_idx

    def normalise_frame(current_frame):
        max_intensity_events = 16
        return (current_frame + max_intensity_events)/float(2*max_intensity_events)
    
    completed_list = False
    start_time = time_list[0]
    next_time = start_time + delta_time
    start_idx, end_idx = 0, num_events
    reconstructed_frames = []
    while not completed_list: 
        start_idx, end_idx = search_duration_idx(time_list[this_frame_idx:], start_time, next_time)
        if end_idx >= num_events - 1:
            completed_list = True
            end_idx = num_events - 1
        current_event_list = events_list[start_idx:end_idx]
        current_frame = accumulate_event_frame(current_event_list, histrange, height, width)
        video_frame = 255*normalise_frame(current_frame).astype(np.uint8)
        reconstructed_frames.append(video_frame)
        start_time, next_time = next_time, next_time + delta_time
    return np.array(reconstructed_frames)