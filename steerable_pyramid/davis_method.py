import os
import re
import shutil
import time
import math
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import utilities.utility_spectrogram as metrics

#################################################################
# rcosFn.m : raised cosine filter
#################################################################
def rcosFn(width, position, values):
    sz = 256
    X = np.pi * np.arange(-sz-1, 2, 1) / (2*sz)
    Y = values[0] + (values[1]-values[0]) * np.cos(X)**2
    Y[0] = Y[1]
    Y[sz+2] = Y[sz+1]
    X = position + (2*width/np.pi) * (X + np.pi/4)
    return X, Y

#################################################################
# pointOp.m : interpolation function
#################################################################
def pointOp(im, lut, origin, increment):
    size = lut.shape[0]
    X = origin + increment*np.arange(0, size, 1)
    Y = lut
    interp_func = interp1d(X, Y, kind='linear', fill_value='extrapolate')
    res = interp_func(im.flatten())
    return res.reshape(im.shape)

#################################################################
# buildSCFpyrLevs.m : perform steerable pyramids
#################################################################
def buildSCFpyrLevs(lodft, log_rad, Xrcos, Yrcos, angle, ht, nbands):
    if ht <= 0:
        lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
        pyr = np.real(lo0).flatten()
        pind = lo0.shape
        return pyr, pind
    else:
        bands = np.zeros((np.prod(lodft.shape), nbands))
        bind = np.zeros((nbands, 2))

        Xrcos = Xrcos - np.log2(2)
        lutsize = 1024
        Xarr = np.arange(-(2*lutsize+1), (lutsize+2), 1)
        Xcosn = np.pi*Xarr/lutsize
        order = nbands-1
        const = (2**(2*order))*(math.factorial(order)**2)/(nbands*math.factorial(2*order))

        alfa = np.mod(np.pi+Xcosn, 2*np.pi) - np.pi
        Ycosn = 2*np.sqrt(const)*(np.cos(Xcosn)**order)*(np.abs(alfa)<np.pi/2)

        himask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0])
        for b in range(nbands):
            b = b+1
            anglemask = pointOp(angle, Ycosn, Xcosn[0]+np.pi*(b-1)/nbands, Xcosn[1]-Xcosn[0])
            banddft = ((-1j)**(nbands-1))*lodft*anglemask*himask
            band = np.fft.ifft2(np.fft.ifftshift(banddft))

            bands[:, (b-1)] = band.flatten()
            bind[(b-1), :] = np.array(band.shape)
        
        dims = np.array(lodft.shape)
        ctr = np.ceil((dims+0.5)/2).astype(int)
        lodims = np.ceil((dims-0.5)/2).astype(int)
        loctr = np.ceil((lodims+0.5)/2).astype(int)
        lostart = ctr-loctr+1
        loend = lostart+lodims-1

        log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
        angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
        lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
        YIrcos = np.abs(np.sqrt(1.0-Yrcos**2))
        lomask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0])

        lodft = lomask*lodft

        npyr, nind = buildSCFpyrLevs(lodft, log_rad, Xrcos, Yrcos, angle, ht-1, nbands)
        pyr = np.concatenate((bands.flatten(), npyr))
        pind = np.vstack((bind, nind))
        return pyr, pind

#################################################################
# buildSCFpyr.m : apply steerable pyramids
#################################################################
def buildSCFpyr(im, ht, order):
    twidth = 1
    nbands = order+1

    dims = np.array(im.shape)
    ctr = np.ceil((dims+0.5)/2).astype(np.int32)

    xx = np.arange(1, dims[1]+1, 1)
    yy = np.arange(1, dims[0]+1, 1)
    xramp, yramp = np.meshgrid((xx-ctr[1])/(dims[1]/2), (yy-ctr[0])/(dims[0]/2))
    angle = np.arctan2(yramp, xramp)
    log_rad = np.sqrt(xramp**2 + yramp**2)
    log_rad[log_rad == 0] = 1e-10
    log_rad = np.log2(log_rad)

    Xrcos, Yrcos = rcosFn(twidth, (-twidth/2), [0, 1])
    Yrcos = np.sqrt(Yrcos)

    YIrcos = np.sqrt(1.0-Yrcos**2)
    lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0])
    imdft = np.fft.fftshift(np.fft.fft2(im))
    lo0dft = imdft*lo0mask

    pyr, pind = buildSCFpyrLevs(lo0dft, log_rad, Xrcos, Yrcos, angle, ht, nbands)

    hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0])
    hi0dft = imdft*hi0mask
    hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))

    real_hi0_flat = np.real(hi0).flatten()
    pyr = np.concatenate((real_hi0_flat, pyr))
    hi0_size = np.array(hi0.shape)
    pind = np.vstack((hi0_size, pind))
    return pyr, pind

#################################################################
# pyrBandIndices.m : find indices to extract pyramid output
#################################################################
def pyrBandIndices(pind, band):
    ind = 1
    for l in range(band-1):
        ind = ind + np.prod(pind[l-1, :])
    indices = np.arange(ind, ind+np.prod(pind[band-1, :])).astype(int)
    return indices

#################################################################
# pyrBand.m : extract pyramid output 
#################################################################
def pyrBand(pyr, pind, band):
    indices = pyrBandIndices(pind, band)
    band_data = pyr[indices]
    res = band_data.reshape((int(pind[band - 1, 0]), int(pind[band - 1, 1])))
    return res

#################################################################
# vmAlignAToB.m : align signals in time  of each pyramid filter
#################################################################
def vmAlignAToB(Ax, Bx):
    acorb = np.convolve(Ax, Bx[::-1], mode='full')
    maxval = np.max(acorb)
    maxind = np.argmax(acorb)
    shiftam = len(Bx) - maxind
    AXout = np.roll(Ax, shiftam)
    return AXout, shiftam

#################################################################
# vmSoundFromVideo.m : carry out the entire process
#################################################################
def vmSoundFromVideo(video, nscalesin, norientationsin, framerate):
    tic = time.time()
    startTime = time.time()

    nScales = nscalesin
    nOrients = norientationsin
    numFramesIn = video.shape[0]
    samplingrate = framerate
    
    # reading first frame of video
    colorframe = video[0]
    refFrame = colorframe

    h, w = refFrame.shape

    nF = int(numFramesIn)
    pyrRef, pind = buildSCFpyr(refFrame, nScales, nOrients-1)
    
    totalsigs = nScales*nOrients
    signalffs = np.zeros((nScales, nOrients, nF))
    ampsigs = np.zeros((nScales, nOrients, nF))

    for q in range(nF):
        if np.mod(q+1, np.floor(nF/100))==1:
            progress = (q+1)/nF
            currentTime = time.time()
            print(f'Progress: {progress*100}% done after {currentTime-startTime} seconds.')
        
        vframein = video[q]
        im = vframein

        pyr, _ = buildSCFpyr(im, nScales, nOrients-1)
        pyrAmp = np.abs(pyr)
        pyrDeltaPhase = np.mod(np.pi+np.angle(pyr)-np.angle(pyrRef), 2*np.pi) - np.pi

        for j in range(nScales):
            bandIdx = 1 + (j)*nOrients + 1
            curH = pind[bandIdx-1, 0]
            curW = pind[bandIdx-1, 1]
            for k in range(nOrients):
                bandIdx = 1 + (j)*nOrients + (k+1)
                amp = pyrBand(pyrAmp, pind, bandIdx)
                phase = pyrBand(pyrDeltaPhase, pind, bandIdx)
                
                phasew = phase*(np.abs(amp)**2)

                sumamp = np.sum(np.abs(amp))
                ampsigs[j, k, q] = sumamp

                mean_phasew = np.mean(phasew.ravel())
                if sumamp == 0 or not np.isfinite(sumamp):
                     signalffs[j, k, q] = 0
                else:
                    signalffs[j, k, q] = mean_phasew/sumamp
    
    sigOut = np.zeros(nF)
    for q in range(nScales):
        for p in range(nOrients):
            sigaligned, _ = vmAlignAToB(np.squeeze(signalffs[q, p, :]), np.squeeze(signalffs[0, 0, :]))
            sigOut = sigOut + sigaligned

    highpassfc = 0.05
    b, a = butter(3, highpassfc, btype='high', analog=False)
    S_x = filtfilt(b, a, sigOut)

    S_x[0:9] = np.mean(S_x)
    maxsx = np.max(S_x)
    minsx = np.min(S_x)
    if maxsx!=1.0 or minsx!=-1.0:
        range_val = maxsx - minsx
        S_x = 2*S_x/range_val
        newmx = np.max(S_x)
        offset = newmx-1.0
        S_x = S_x-offset
    
    if not np.isfinite(sigOut).all():
        sigOut = np.nan_to_num(sigOut)
    if not np.isfinite(S_x).all():
        S_x = np.nan_to_num(S_x) 
    metrics.show_spectrogram(sigOut, samplingrate, 80, 40, 'Unfiltered')
    metrics.show_spectrogram(S_x, samplingrate, 80, 40, 'Filtered')

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(sigOut)
    ax[0].set_title('Unfiltered Signal Plot')
    ax[1].plot(S_x)
    ax[1].set_title('Filtered Signal Plot')
    plt.show()

    sd.play(S_x, samplerate=2200)
    sd.wait()
    return S_x

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def ebvmSoundfromVideo(audiosavepath, nscalesin, norientationsin, framerate):
    desktop_path = os.path.join(os.path.expanduser("~"), "Documents")
    folder_path = os.path.join(desktop_path, 'TemporaryEventVideoSegments')
    video_files = None
    for _, _, files in os.walk(folder_path):
        video_files = [file for file in files if not file.startswith('.')]
        break
    video_files.sort(key=extract_number)
    num_files = len(video_files)

    new_folder_path = os.path.join(desktop_path, 'TemporarySoundSegments')
    try:
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Temporary folder created: {new_folder_path}")
    except Exception as e:
        print(f"Error creating temporary folder: {e}")
    
    segment_count = 0
    for file in video_files:
        segment_count += 1
        file_path = os.path.join(folder_path, file)
        save_path = os.path.join(new_folder_path, f'Segment{segment_count}.wav')
        frames = np.load(file_path)
        signal = vmSoundFromVideo(frames, nscalesin, norientationsin, framerate)
        sf.write(save_path, signal, framerate)
        print(f"Converted and saved audio segment {segment_count}/{num_files} to Documents.")
        os.remove(file_path)
    os.rmdir(folder_path)

    for _, _, files in os.walk(new_folder_path):
        audio_files = [file for file in files if not file.startswith('.')]
        break
    audio_files.sort(key=extract_number)

    audio = []
    for file in audio_files:
        file_path = os.path.join(new_folder_path, file)
        audio_data, _ = sf.read(file_path)
        audio.extend(audio_data)
    sf.write(audiosavepath, audio, framerate)
    print(f'Saved final audio to {audiosavepath}.')
    shutil.rmtree(new_folder_path)
    