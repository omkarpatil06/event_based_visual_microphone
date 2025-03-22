import os
import cv2
import time
import math
import numpy as np
from scipy.interpolate import interp1d

def pointOp(im, lut, origin, increment):
    # length of the interpolation LUT
    N = lut.shape[0]
    # f(x) and x for creating a linear mapping
    Y = lut
    X = origin + increment*np.arange(0, N, 1)
    interp_func = interp1d(X, Y, kind='linear', fill_value='extrapolate')
    # use interpolation function on frame
    res = interp_func(im.flatten())
    return res.reshape(im.shape)

def rcosFn(width, position, max_min):
    N = 256
    # x values from -pi/2, 0
    x = np.pi * np.arange(-N-1, 1+1, 1) / (2*N)
    Y = max_min[0] + (max_min[1]-max_min[0]) * np.cos(x)**2
    Y[0], Y[-1] = Y[1], Y[-2]
    # x values from -1, 0
    X = position + (2*width/np.pi) * (x + np.pi/4)
    return X, Y

def buildSCFpyr(im, ht, order):
    nbands = order+1
    # image dimensions and centre
    dims = np.array(im.shape)
    ctr = np.ceil((dims+0.5)/2).astype(np.int32)
    # radius and angle mask
    xx = np.arange(1, dims[1]+1, 1)
    yy = np.arange(1, dims[0]+1, 1)
    xramp, yramp = np.meshgrid((xx-ctr[1])/(dims[1]/2), (yy-ctr[0])/(dims[0]/2))
    angle = np.arctan2(yramp, xramp)
    log_rad = np.sqrt(xramp**2 + yramp**2)
    log_rad[log_rad == 0] = 1e-10
    log_rad = np.log2(log_rad)

    # low-pass filter
    X, Y = rcosFn(1, (-1/2), [0, 1])
    Y_low = np.sqrt(1.0-Y)
    lo0mask = pointOp(log_rad, Y_low, X[0], X[1]-X[0])
    # high-pass filter
    Y_high = np.sqrt(Y)
    hi0mask = pointOp(log_rad, Y_high, X[0], X[1]-X[0])
    # low-pass input to pyramid
    imdft = np.fft.fftshift(np.fft.fft2(im))
    lo0dft = imdft*lo0mask
    hi0dft = imdft*hi0mask

    # pass low-pass input to the steerable pyramid
    pyr, pind = buildSCFpyrLevs(lo0dft, log_rad, X, Y_high, angle, ht, nbands)

    # high-pass addition to the pyramid outputs
    hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))
    real_hi0_flat = np.real(hi0).flatten()
    pyr = np.concatenate((real_hi0_flat, pyr))
    hi0_size = np.array(hi0.shape)
    pind = np.vstack((hi0_size, pind))
    return pyr, pind

def buildSCFpyrLevs(lodft, log_rad, X, Y_high, angle, ht, nbands):
    if ht <= 0:
        lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
        pyr = np.real(lo0).flatten().astype(np.complex64)
        pind = lo0.shape
        return pyr, pind
    else:
        # store pyramid output
        bands = np.zeros((np.prod(lodft.shape), nbands)).astype(np.complex64)
        bind = np.zeros((nbands, 2))
        # create and apply angle masks
        N = 1024
        O = nbands-1
        X = X - np.log2(2)
        Xcosn = np.pi * np.arange(-(2*N+1), (N+2), 1) / N
        const = (2**(2*O))*(math.factorial(O)**2)/(nbands*math.factorial(2*O))
        alfa = np.mod(np.pi+Xcosn, 2*np.pi) - np.pi
        Ycosn = 2*np.sqrt(const)*(np.cos(Xcosn)**O)*(np.abs(alfa)<np.pi/2)
        himask = pointOp(log_rad, Y_high, X[0], X[1]-X[0])
        for b in range(nbands):
            b = b+1
            anglemask = pointOp(angle, Ycosn, Xcosn[0]+np.pi*(b-1)/nbands, Xcosn[1]-Xcosn[0])
            banddft = ((-1j)**(nbands-1))*lodft*anglemask*himask
            band = np.fft.ifft2(np.fft.ifftshift(banddft)).astype(np.complex64)
            bands[:, (b-1)] = band.flatten()
            bind[(b-1), :] = np.array(band.shape)
        
        # update step for next recursive step
        dims = np.array(lodft.shape)
        ctr = np.ceil((dims+0.5)/2).astype(int)
        lodims = np.ceil((dims-0.5)/2).astype(int)
        loctr = np.ceil((lodims+0.5)/2).astype(int)
        lostart = ctr-loctr+1
        loend = lostart+lodims-1
        log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
        angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
        lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
        Y_low = np.abs(np.sqrt(1.0-Y_high**2))
        lomask = pointOp(log_rad, Y_low, X[0], X[1]-X[0])
        lodft = lomask*lodft

        # recursive step
        npyr, nind = buildSCFpyrLevs(lodft, log_rad, X, Y_high, angle, ht-1, nbands)
        pyr = np.concatenate((bands.flatten(order='F'), npyr))
        pind = np.vstack((bind, nind))
        return pyr, pind

def pyrBandIndices(pind, band):
    ind = 0
    for l in range(band-1):
        ind = ind + np.prod(pind[l, :])
    indices = np.arange(ind, ind+np.prod(pind[band, :])).astype(int)
    return indices

def pyrBand(pyr, pind, band):
    indices = pyrBandIndices(pind, band)
    band_data = pyr[indices]
    res = band_data.reshape((int(pind[band, 0]), int(pind[band, 1])))
    return res

def vmVideoPyramid(video, nscalesin, norientationsin):
    startTime = time.time()

    # pyramid properties 
    nScales = nscalesin
    nOrients = norientationsin
    numFramesIn = video.shape[0]
    nF = int(numFramesIn)
    
    # reading first frame of video
    refFrame = video[0]    
    pyrRef, pind = buildSCFpyr(refFrame, nScales, nOrients-1)
    
    # saving the frames
    video_matrix = [[[] for _ in range(nOrients)] for _ in range(nScales)]
    for q in range(nF):
        if np.mod(q+1, np.floor(nF/100))==1:
            progress = (q+1)/nF
            currentTime = time.time()
            print(f'Progress: {progress*100:.3f}% done after {currentTime-startTime:.3f} seconds.')
    
        # reading current frame of video
        im = video[q]
        pyr, _ = buildSCFpyr(im, nScales, nOrients-1)
        # reading and storing the pyramid outputs
        pyrDeltaPhase = np.mod(np.pi+np.angle(pyr)-np.angle(pyrRef), 2*np.pi) - np.pi
        for j in range(nScales):
            bandIdx = 1 + (j)*nOrients + 1
            for k in range(nOrients):
                bandIdx = 1 + (j)*nOrients + k
                phase = pyrBand(pyrDeltaPhase, pind, bandIdx)
                print(phase.shape)
                video_matrix[j][k].append(np.array(phase).astype(np.float32))
    return video_matrix

def load_video(video_path):
    video_frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = gray_frame.astype(np.float32) / 255.0
        video_frames.append(gray_frame)
    cap.release()
    video_frames = np.array(video_frames)
    return video_frames

def save_pyramids(source_folder, target_folder, nscalesin, norientationsin):
    # ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)
    # iterate through all video files in the source folder
    for video_file in os.listdir(source_folder):
        # skip non-video files
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue
        # get the video file path
        video_path = os.path.join(source_folder, video_file)
        video_frames = load_video(video_path)
        # create a subfolder in the target folder with the video name (without extension)
        video_name = os.path.splitext(video_file)[0]
        subfolder_path = os.path.join(target_folder, video_name)
        os.makedirs(subfolder_path, exist_ok=True)
        # determine all frames from pyramid
        print(f"Processing video: {video_file}")
        print(f"Saving frames to: {subfolder_path}")
        video_matrix = vmVideoPyramid(video_frames, nscalesin, norientationsin)
        for j in range(nscalesin):
            for k in range(norientationsin):
                file_name = os.path.join(subfolder_path, f"{video_name}_{j}_{k}.npy")
                np.save(file_name, np.array(video_matrix[j][k]))
                print(f"Saved: {file_name}, Video dim: {np.array(video_matrix[j][k]).shape}")
        print(f"Finished processing {video_file}, saved {video_frames.shape[0]} frames.\n")

# Example usage
source_folder = "/Volumes/Omkar 5T/evbm/haoqi_roi/"  # Replace with the path to your folder containing videos
target_folder = "/Volumes/Omkar 5T/evbm/haoqi_pyramid/"  # Replace with the path to the target folder

save_pyramids(source_folder, target_folder, 2, 2)
