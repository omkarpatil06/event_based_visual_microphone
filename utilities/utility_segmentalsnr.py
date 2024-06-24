import numpy as np

def segmental_snr(original, reconstructed, segment_length=512, overlap=0.5):
    step = int(segment_length * (1 - overlap))
    seg_snr = []
    # Ensure that both signals are numpy arrays
    original = np.array(original)
    reconstructed = np.array(reconstructed)
    # Calculate SNR for each segment
    for start in range(0, len(original) - segment_length + 1, step):
        end = start + segment_length
        orig_segment = original[start:end]
        rec_segment = reconstructed[start:end]
        # Calculate the noise segment
        noise = orig_segment - rec_segment
        # Calculate signal energy and noise energy
        signal_energy = np.sum(orig_segment ** 2)
        noise_energy = np.sum(noise ** 2)
        # Avoid division by zero
        if noise_energy == 0:
            noise_energy = 1e-10
        # Calculate SNR for this segment
        snr = 10 * np.log10(signal_energy / noise_energy)
        seg_snr.append(snr)
    snr_value = np.mean(seg_snr)
    print("Segmental SNR: {:.2f} dB".format(snr_value))
    return snr_value