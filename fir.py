# fir.py

import numpy as np
import scipy.signal as signal

SAMPLE_RATE = 44100  # Standard audio sample rate

# Define filter frequency bands
bass_band = [20, 250]  # Bass frequencies
mid_band = [250, 4000]  # Mid frequencies
treble_band = [4000, 20000]  # Treble frequencies

def create_fir_filter(gain, band, numtaps=101, sample_rate=SAMPLE_RATE, pass_zero='bandpass'):
    """
    Create a FIR filter for a given band and gain.
    """
    nyquist_freq = 0.5 * sample_rate
    lowcut, highcut = band
    lowcut_norm = lowcut / nyquist_freq
    highcut_norm = highcut / nyquist_freq

    filter_coefs = signal.firwin(numtaps, [lowcut_norm, highcut_norm], pass_zero=pass_zero)
    return gain * filter_coefs

def apply_filters(audio_data, bass_gain, mid_gain, treble_gain):
    """
    Apply FIR filters based on slider values.
    """
    # Create filters for each frequency band
    bass_filter = create_fir_filter(bass_gain, bass_band)
    mid_filter = create_fir_filter(mid_gain, mid_band)
    treble_filter = create_fir_filter(treble_gain, treble_band)

    if audio_data.ndim == 1:  # Mono
        bass_filtered = signal.convolve(audio_data, bass_filter, mode="same")
        mid_filtered = signal.convolve(audio_data, mid_filter, mode="same")
        treble_filtered = signal.convolve(audio_data, treble_filter, mode="same")
        filtered_audio = bass_filtered + mid_filtered + treble_filtered
        return filtered_audio, bass_filtered, mid_filtered, treble_filtered

    elif audio_data.ndim == 2:  # Stereo
        # Apply filter to each channel separately
        bass_filtered_left = signal.convolve(audio_data[:, 0], bass_filter, mode="same")
        mid_filtered_left = signal.convolve(audio_data[:, 0], mid_filter, mode="same")
        treble_filtered_left = signal.convolve(audio_data[:, 0], treble_filter, mode="same")

        bass_filtered_right = signal.convolve(audio_data[:, 1], bass_filter, mode="same")
        mid_filtered_right = signal.convolve(audio_data[:, 1], mid_filter, mode="same")
        treble_filtered_right = signal.convolve(audio_data[:, 1], treble_filter, mode="same")

        # Combine filtered channels for left and right
        filtered_audio_left = bass_filtered_left + mid_filtered_left + treble_filtered_left
        filtered_audio_right = bass_filtered_right + mid_filtered_right + treble_filtered_right

        # Stack to create a stereo output
        filtered_audio = np.stack((filtered_audio_left, filtered_audio_right), axis=-1)
        bass_filtered = np.stack((bass_filtered_left, bass_filtered_right), axis=-1)
        mid_filtered = np.stack((mid_filtered_left, mid_filtered_right), axis=-1)
        treble_filtered = np.stack((treble_filtered_left, treble_filtered_right), axis=-1)

        return filtered_audio, bass_filtered, mid_filtered, treble_filtered

    else:
        raise ValueError("Unsupported audio data dimensionality")
