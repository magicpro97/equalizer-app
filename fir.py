import numpy as np
from scipy.signal import lfilter, firwin

SAMPLE_RATE = 44100  # Default sample rate

def apply_filters(audio_data, bass_gain, mid_gain, treble_gain):
    error_values = [np.inf, -np.inf, np.nan]
    if bass_gain in error_values or mid_gain in error_values or treble_gain in error_values:
        raise ValueError("Gain values cannot be infinite, negative infinity, or NaN")
    
    # Add input validation
    if audio_data is None:
        raise TypeError("Audio data cannot be None")

    if audio_data.ndim > 2:
        raise ValueError("Audio data must be either mono (1D) or stereo (2D with channels as columns)")
    
    if audio_data.ndim == 2 and audio_data.shape[1] != 2:
        raise ValueError("Stereo audio data must have channels as columns (shape should be (samples, 2))")
    
    # Handle empty array case
    if len(audio_data) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    bass_filtered = apply_bandpass_filter(audio_data, 20, 250, bass_gain)
    mid_filtered = apply_bandpass_filter(audio_data, 250, 4000, mid_gain)
    treble_filtered = apply_bandpass_filter(audio_data, 4000, 20000, treble_gain)

    filtered_audio = bass_filtered + mid_filtered + treble_filtered
    return filtered_audio, bass_filtered, mid_filtered, treble_filtered

def apply_bandpass_filter(data, lowcut, highcut, gain, fs=SAMPLE_RATE, numtaps=101, apply_gain=True):
    # Add input validation
    if lowcut is None or highcut is None:
        raise TypeError("Band frequencies cannot be None")
    
    if numtaps <= 0:
        raise ValueError("Number of taps must be positive")
    
    if lowcut >= highcut:
        raise ValueError("Band frequencies must be in ascending order")
    
    # Handle empty array case
    if len(data) == 0:
        return np.array([])
    
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
