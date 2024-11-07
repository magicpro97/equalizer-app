import librosa
import numpy as np
import scipy.signal as signal

SAMPLE_RATE = 44100  # Standard audio sample rate
BLOCK_SIZE = 2048  # Block size for processing (small size for faster processing)
DEFAULT_NUMTAPS = 32  # Shorter filter length for faster processing

# Define filter frequency bands
bass_band = [20, 250]  # Bass frequencies
mid_band = [250, 4000]  # Mid frequencies
treble_band = [4000, 20000]  # Treble frequencies


def create_fir_filter(gain, band, numtaps=DEFAULT_NUMTAPS, sample_rate=SAMPLE_RATE):
    nyquist_freq = 0.5 * sample_rate
    lowcut, highcut = band
    lowcut_norm = lowcut / nyquist_freq
    highcut_norm = highcut / nyquist_freq

    filter_coefs = gain * signal.firwin(numtaps, [lowcut_norm, highcut_norm], pass_zero=False)
    return filter_coefs


def process_block(audio_block, filters):
    """Apply filters to a single block of audio data."""
    bass_filter, mid_filter, treble_filter = filters
    bass_filtered = signal.fftconvolve(audio_block, bass_filter, mode="same")
    mid_filtered = signal.fftconvolve(audio_block, mid_filter, mode="same")
    treble_filtered = signal.fftconvolve(audio_block, treble_filter, mode="same")
    return bass_filtered, mid_filtered, treble_filtered


def apply_filters(audio_data, bass_gain, mid_gain, treble_gain):
    bass_filter = create_fir_filter(bass_gain, bass_band)
    mid_filter = create_fir_filter(mid_gain, mid_band)
    treble_filter = create_fir_filter(treble_gain, treble_band)

    filters = (bass_filter, mid_filter, treble_filter)

    # Initialize arrays to store each filtered output
    bass_filtered_output = np.zeros_like(audio_data)
    mid_filtered_output = np.zeros_like(audio_data)
    treble_filtered_output = np.zeros_like(audio_data)

    # Process in blocks for memory efficiency and faster playback
    for start in range(0, len(audio_data), BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, len(audio_data))
        audio_block = audio_data[start:end]

        # Handle mono and stereo cases
        if audio_data.ndim == 1:  # Mono
            bass_block, mid_block, treble_block = process_block(audio_block, filters)
            bass_filtered_output[start:end] = bass_block
            mid_filtered_output[start:end] = mid_block
            treble_filtered_output[start:end] = treble_block

        elif audio_data.ndim == 2:  # Stereo
            bass_block_left, mid_block_left, treble_block_left = process_block(audio_block[:, 0], filters)
            bass_block_right, mid_block_right, treble_block_right = process_block(audio_block[:, 1], filters)

            bass_filtered_output[start:end, 0] = bass_block_left
            mid_filtered_output[start:end, 0] = mid_block_left
            treble_filtered_output[start:end, 0] = treble_block_left

            bass_filtered_output[start:end, 1] = bass_block_right
            mid_filtered_output[start:end, 1] = mid_block_right
            treble_filtered_output[start:end, 1] = treble_block_right

    return bass_filtered_output + mid_filtered_output + treble_filtered_output, bass_filtered_output, mid_filtered_output, treble_filtered_output

def speed_up_audio(audio_data, speed_factor=2.0):
    """
    Speed up audio by a specified factor without changing pitch.

    Parameters:
    - audio_data: np.ndarray, audio waveform
    - speed_factor: float, factor by which to speed up (e.g., 2.0 to double speed)

    Returns:
    - sped_up_audio: np.ndarray, sped-up audio waveform
    """
    # Apply time-stretching
    sped_up_audio = librosa.effects.time_stretch(audio_data, rate=speed_factor)
    return sped_up_audio