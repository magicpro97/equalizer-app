import numpy as np
from scipy.signal import lfilter, firwin

SAMPLE_RATE = 44100  # Default sample rate

def apply_filters(audio_data, bass_gain, mid_gain, treble_gain):
    bass_filtered = apply_bandpass_filter(audio_data, 20, 250, bass_gain)
    mid_filtered = apply_bandpass_filter(audio_data, 250, 4000, mid_gain)
    treble_filtered = apply_bandpass_filter(audio_data, 4000, 20000, treble_gain)

    filtered_audio = bass_filtered + mid_filtered + treble_filtered
    return filtered_audio, bass_filtered, mid_filtered, treble_filtered

def apply_bandpass_filter(data, lowcut, highcut, gain, fs=SAMPLE_RATE, numtaps=101, apply_gain=True):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    taps = firwin(numtaps, [low, high], pass_zero=False)
    filtered_data = lfilter(taps, 1.0, data)
    if apply_gain:
        filtered_data *= gain
    return filtered_data

def speed_up_audio(audio_data, factor):
    indices = np.round(np.arange(0, len(audio_data), factor)).astype(int)
    indices = indices[indices < len(audio_data)]
    return audio_data[indices]