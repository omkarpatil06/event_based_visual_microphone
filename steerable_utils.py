import numpy as np
from scipy.interpolate import interp1d
from scipy.special import factorial
from sklearn.preprocessing import MinMaxScaler

#################################################################
# BUILDING STEERABLE PYRAMID UTILITY FUNCTIONS
#################################################################
def interpolate(frame, lut, origin, increment):
    x = origin + increment*np.arange(0, len(lut))
    func_interp = interp1d(x, lut, kind='linear', fill_value='extrapolate')
    return func_interp(frame.flatten()).reshape(frame.shape)

# def build_pyramid(angle, log_radian, rcos_w, rcos_h, num_scales, num_steerable_filters):
#     if num_scales <= 0:
#         return None, None
#     else:
#         # generate steerable filter masks
#         rcos_w = rcos_w - np.log2(2)
#         cosn_w = np.pi/1024*np.arange(-2025, 1026)
#         order = num_steerable_filters-1
#         constant = (2**(2*order))*(factorial(order)**2)/(num_steerable_filters*factorial(2*order))
#         alpha = ((np.pi + cosn_w) % (2*np.pi)) - np.pi
#         cosn_h = 2*np.sqrt(constant)*(np.cos(cosn_w)**order)*(np.abs(alpha)<np.pi/2)
#         hp_mask = interpolate(frame=log_radian, lut=rcos_h, origin=rcos_w[0], increment=rcos_w[1]-rcos_w[0])
#         filters = []
#         filter_size = []
#         for filter_idx in range(num_steerable_filters):
#             angle_mask = ((-1j)**(num_steerable_filters-1))*hp_mask*interpolate(angle, cosn_h, cosn_w[0]+np.pi*(filter_idx-1)/num_steerable_filters, cosn_w[1]-cosn_w[0])
#             filters.extend(angle_mask.flatten())
#             angle_mask_shape = np.array([angle_mask.shape[0], angle_mask.shape[1]])
#             filter_size.append(angle_mask_shape)
#         # downsampling for next scale
#         height, width = angle.shape
#         centre_h, centre_w = np.ceil((height+0.5)/2).astype(int), np.ceil((width+0.5)/2).astype(int)
#         lheight, lwidth = np.ceil((height-0.5)/2).astype(int), np.ceil((width-0.5)/2).astype(int)
#         centre_lh, centre_lw = np.ceil((lheight+0.5)/2).astype(int), np.ceil((lwidth+0.5)/2).astype(int)
#         start_lh, start_lw = centre_h-centre_lh+1, centre_w-centre_lw+1
#         end_lh, end_lw = start_lh+lheight-1, start_lw+lwidth-1
#         angle = angle[start_lh:end_lh, start_lw:end_lw]
#         log_radian = log_radian[start_lh:end_lh, start_lw:end_lw]
#         filter, filtersize = build_pyramid(angle, log_radian, rcos_w, rcos_h, num_scales-1, num_steerable_filters)
#         filters.extend(filter) if filter is not None else filters
#         filter_size.extend(filtersize) if filtersize is not None else filter_size
#         return filters, filter_size

def apply_steerable_filter(frame_lpdft, log_radian, xrcos, yrcos, angle, num_scales, num_steerable_filters):
    if num_scales <= 0:
        return None, None
    else:
        # preparation for creating steerable filters
        xrcos = xrcos - np.log2(2)
        xcosn = np.pi/1024*np.arange(-2025, 1026)
        order = num_steerable_filters-1
        constant = (2**(2*order))*(factorial(order)**2)/(num_steerable_filters*factorial(2*order))
        alpha = np.mod(np.pi + xcosn, 2*np.pi) - np.pi
        ycosn = 2*np.sqrt(constant)*(np.cos(xcosn)**order)*(np.abs(alpha)<np.pi/2)

        # apply steerable filter to frame at a scale
        band = []
        band_size = []
        hp_mask = interpolate(frame=log_radian, lut=yrcos, origin=xrcos[0], increment=xrcos[1]-xrcos[0])
        for filter_idx in range(num_steerable_filters):
            b = filter_idx+1
            angle_mask = interpolate(frame=angle, lut=ycosn, origin=xcosn[0]+np.pi*(b-1)/num_steerable_filters, increment=xcosn[1]-xcosn[0])
            band_dft = ((-1j)**(num_steerable_filters-1))*frame_lpdft*angle_mask*hp_mask
            band_idft = np.fft.ifft2(np.fft.ifftshift(band_dft))
            band.extend(band_idft.flatten())
            band_shape = np.array([band_idft.shape[0], band_idft.shape[1]])
            band_size.append(band_shape)

        # preperation for subsampling for next stage in pyramid
        height, width = frame_lpdft.shape
        centre_h, centre_w = np.ceil((height+0.5)/2).astype(int), np.ceil((width+0.5)/2).astype(int)
        lheight, lwidth = np.ceil((height-0.5)/2).astype(int), np.ceil((width-0.5)/2).astype(int)
        centre_lh, centre_lw = np.ceil((lheight+0.5)/2).astype(int), np.ceil((lwidth+0.5)/2).astype(int)
        start_lh, start_lw = centre_h-centre_lh+1, centre_w-centre_lw+1
        end_lh, end_lw = start_lh+lheight-1, start_lw+lwidth-1

        # subsampling for next stage in pyramid
        log_radian = log_radian[start_lh:end_lh, start_lw:end_lw]
        angle = angle[start_lh:end_lh, start_lw:end_lw]
        frame_lpdft = frame_lpdft[start_lh:end_lh, start_lw:end_lw]
        yircos = np.abs(np.sqrt(1-yrcos**2))
        lp_mask = interpolate(frame=log_radian, lut=yircos, origin=xrcos[0], increment=xrcos[1]-xrcos[0])
        frame_lpdft = lp_mask*frame_lpdft
        bands, band_sizes = apply_steerable_filter(frame_lpdft, log_radian, xrcos, yrcos, angle, num_scales-1, num_steerable_filters)

        # return for recursive algorithm
        band = bands + band if bands is not None else band
        band_size = band_sizes + band_size if band_sizes is not None else band_size
        return band, band_size

#################################################################
# BUILD STEERABLE PYRAMID FILTERS FUNCTION
#################################################################
def pyramid_output(frame, num_scales, num_steerable_filters):
    # Trying to create a unit frame from +/-1
    height, width = frame.shape
    centre_h, centre_w = np.ceil((height+0.5)/2).astype(int), np.ceil((width+0.5)/2).astype(int)
    h = ((np.arange(1, height+1, 1) - centre_h)/(height/2)).astype(float)
    w = ((np.arange(1, width+1, 1) - centre_w)/(width/2)).astype(float)
    hh, ww = np.meshgrid(w, h)

    # Converting the unit frame to polar coordinates
    argument = np.arctan2(hh, ww)
    radian = np.sqrt(ww**2 + hh**2)
    zero_idx = np.where(radian == 0)
    radian[zero_idx[0][0], zero_idx[1][0]] = radian[zero_idx[0][0], zero_idx[1][0]-1]
    log_radian = np.log2(radian)

    # create raised cosine filter low pass and high pass filter
    xrcos = np.pi*np.arange(-257, 0, 1)/(512)
    yrcos = np.cos(xrcos)**2
    yrcos[0], yrcos[-1] = yrcos[1], yrcos[-2]
    xrcos = -0.5+(2/np.pi)*(xrcos+np.pi/4)
    # low pass and high pass filter
    yrcos = np.sqrt(yrcos)
    yircos = np.sqrt(1 - yrcos**2)
    lp_mask = interpolate(frame=log_radian, lut=yircos, origin=xrcos[0], increment=xrcos[1]-xrcos[0])

    # applying low-pass mask to frame
    frame_dft = np.fft.fftshift(np.fft.fft2(frame))
    frame_lpdft = frame_dft*lp_mask

    # creates complex steerable filters
    filters, filter_size = apply_steerable_filter(frame_lpdft, log_radian, xrcos, yrcos, argument, num_scales, num_steerable_filters)
    start_idx, end_idx = 0, 0
    steerable_filters = []
    for size in filter_size:
        end_idx = start_idx + np.prod(size).tolist() 
        steerable_filters.append(np.array(filters[start_idx:end_idx]).reshape(size[0], size[1]))
        start_idx = end_idx
    return steerable_filters

#################################################################
# APPLYING STEERABLE FILTERS TO VIDEO FUNCTIONS
#################################################################
# def apply_steerable_pyramid(image, steerable_filters, lp_mask, log_radian, rcos_w, rcos_hstar):
#     num_scales = 4
#     num_steerable_filters = 2
#     image_fft = np.fft.fft2(image)
#     image_fft = np.fft.fftshift(image_fft)
#     lp_image = image_fft*lp_mask
#     rcos_w = rcos_w - np.log2(2)
#     steerable_images = []
#     for _ in range(num_scales):
#         for _ in range(num_steerable_filters):
#             steerable_images.append(lp_image)
#         height, width = lp_image.shape
#         centre_h, centre_w = np.ceil((height+0.5)/2).astype(int), np.ceil((width+0.5)/2).astype(int)
#         lheight, lwidth = np.ceil((height-0.5)/2).astype(int), np.ceil((width-0.5)/2).astype(int)
#         centre_lh, centre_lw = np.ceil((lheight+0.5)/2).astype(int), np.ceil((lwidth+0.5)/2).astype(int)
#         start_lh, start_lw = centre_h-centre_lh+1, centre_w-centre_lw+1
#         end_lh, end_lw = start_lh+lheight-1, start_lw+lwidth-1
#         lp_image = lp_image[start_lh:end_lh, start_lw:end_lw]
#         log_radian = log_radian[start_lh:end_lh, start_lw:end_lw]
#         rcos_w = rcos_w - np.log2(2)
#         lp_mask = interpolate(frame=log_radian, lut=rcos_hstar, origin=rcos_w[0], increment=rcos_w[1]-rcos_w[0])
#         lp_image = lp_image*lp_mask
#     pyramid_response = [steerable_images[idx]*steerable_filters[idx] for idx in range(num_scales*num_steerable_filters)]
#     pyramid_response = [np.fft.ifft2(np.fft.ifftshift(response)) for response in pyramid_response]
#     return pyramid_response

def align_signals(signal):
    ref_signal = signal[0]
    signal_out = signal[0]
    for sig in signal[1:]:
        a_cor_b = np.correlate(sig, ref_signal[::-1], mode='full')
        maxind = np.argmax(a_cor_b)
        shiftam = len(ref_signal) - maxind - 1
        sig_out = np.roll(sig, shiftam)
        signal_out = np.vstack([signal_out, sig_out])
    return signal_out

#################################################################
# VIDEO TO SIGNAL CONVERSION FUNCTION
#################################################################
def video_to_sound(images):
    num_scales = 4
    num_steerable_filters = 2

    # produce initial frame for comparison
    initial_image = images[0]
    initial_steerable_image = pyramid_output(initial_image, num_scales, num_steerable_filters)

    # generate signal based on the equation in paper
    signal = None
    for image in images[1:]:
        steerable_image = pyramid_output(image, num_scales, num_steerable_filters)
        # Amplitude frame of each steerable filter output
        amp_image = [np.abs(image) for image in steerable_image]
        # Phase frame of each steerable filter output
        phase_image = [np.angle(image) for image in steerable_image]
        # Differential phase frame of each steerable filter output
        difference_phase_image = [np.mod(np.pi + phase_image[idx] - np.angle(initial_steerable_image[idx]), 2*np.pi) - np.pi for idx in range(len(phase_image))]
        # Sum of amplitude weighted differental phase frame
        weighted_image = [(np.abs(amp_image[idx])**2)*difference_phase_image[idx] for idx in range(len(phase_image))]
        # scalar value per orientation and scale
        sum_amplitude = [np.sum(np.abs(image)) for image in amp_image]
        sum_weighted_image = np.array([np.mean(weighted_image[idx])/sum_amplitude[idx] for idx in range(len(weighted_image))])
        signal = np.vstack([signal, sum_weighted_image]) if signal is not None else sum_weighted_image
    # Align signal and find mean to find output signal
    signal = signal.T
    signal = align_signals(signal)
    signal = np.mean(signal, axis=0)
    signal = (signal-signal.min())/(signal.max()-signal.min())
    return signal