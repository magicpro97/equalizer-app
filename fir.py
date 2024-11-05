# fir.py

import numpy as np
import scipy.signal as signal

SAMPLE_RATE = 44100  # Standard audio sample rate

# Define filter frequency bands
bass_band = [20, 250]  # Bass frequencies
mid_band = [250, 4000]  # Mid frequencies
treble_band = [4000, 20000]  # Treble frequencies

def create_fir_filter(gain, band, numtaps=101, sample_rate=SAMPLE_RATE):
    """
    Create a FIR filter for a given band and gain.
    """
    nyquist_freq = 0.5 * sample_rate
    lowcut, highcut = band
    lowcut_norm = lowcut / nyquist_freq
    highcut_norm = highcut / nyquist_freq

    filter_coefs = signal.firwin(numtaps, [lowcut_norm, highcut_norm], pass_zero=False)
    return gain * filter_coefs

def apply_filters(audio_data, bass_gain, mid_gain, treble_gain):
    """
    Apply FIR filters based on slider values.
    """
    # Create filters for each frequency band
    bass_filter = create_fir_filter(bass_gain, bass_band)
    mid_filter = create_fir_filter(mid_gain, mid_band)
    treble_filter = create_fir_filter(treble_gain, treble_band)

    # Check if audio_data is stereo or mono
    if audio_data.ndim == 1:  # Mono
        bass_filtered = signal.convolve(audio_data, bass_filter, mode="same")
        mid_filtered = signal.convolve(audio_data, mid_filter, mode="same")
        treble_filtered = signal.convolve(audio_data, treble_filter, mode="same")
        return bass_filtered + mid_filtered + treble_filtered, bass_filtered, mid_filtered, treble_filtered
    elif audio_data.ndim == 2:  # Stereo
        # Apply filter to each channel separately
        bass_filtered = signal.convolve(audio_data[:, 0], bass_filter, mode="same")
        mid_filtered = signal.convolve(audio_data[:, 0], mid_filter, mode="same")
        treble_filtered = signal.convolve(audio_data[:, 0], treble_filter, mode="same")

        # Combine filtered channels
        filtered_audio = np.stack((bass_filtered + mid_filtered + treble_filtered,
                                   bass_filtered + mid_filtered + treble_filtered), axis=-1)
        return filtered_audio, bass_filtered, mid_filtered, treble_filtered
    else:
        raise ValueError("Unsupported audio data dimensionality")
