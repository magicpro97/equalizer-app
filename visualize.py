import librosa
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st
from scipy.signal import butter, lfilter

from fir import SAMPLE_RATE, apply_filters


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq

    # Ensure cutoff frequencies are within valid range
    low = max(0, min(low, 1))
    high = max(0, min(high, 1))

    print(low, high)

    b, a = butter(order, [low, high], btype='bandpass')

    filtered_data = lfilter(b, a, data)
    return filtered_data

# @st.dialog("Visualize dialog")
def visualize_audio_file(audio_bytes, container, bass_gain, mid_gain, treble_gain):
    filtered_audio, bass_filtered, mid_filtered, treble_filtered = apply_filters(audio_bytes, bass_gain, mid_gain,
                                                                                 treble_gain)
    origin_audio = audio_bytes
    visualize_bands(origin_audio, filtered_audio, bass_filtered, mid_filtered, treble_filtered, container)

def clean_audio_data(audio_data):
    audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    return audio_data

def visualize_bands(origin_audio, filtered_audio, bass_filtered, mid_filtered, treble_filtered, container):
    visualize_audio(origin_audio, container, SAMPLE_RATE, title="Original Audio")
    visualize_audio(filtered_audio, container, SAMPLE_RATE, title="Filtered Audio")
    visualize_audio(bass_filtered, container, SAMPLE_RATE, title="Bass Band")
    visualize_audio(mid_filtered, container, SAMPLE_RATE, title="Mid Band")
    visualize_audio(treble_filtered, container, SAMPLE_RATE, title="Treble Band")

def visualize_audio(audio_data, container, sample_rate=SAMPLE_RATE,title="Audio Signal"):
    audio_data = clean_audio_data(audio_data)
    """
    Visualize the audio signal in time and frequency domains.
    """
    # Plot the waveform
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.title(f"Waveform - {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    container.pyplot(plt)
    plt.close()

    # Plot the spectrogram
    plt.figure(figsize=(14, 5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    D = clean_audio_data(D)
    librosa.display.specshow(D, sr=sample_rate, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram - {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    container.pyplot(plt)
    plt.close()

# Archive ------------------------------
def filter_audio(audio_data, sample_rate, freq_range):
  # Implement a simple band-pass filter using FFT
  fft_spectrum = np.fft.fft(audio_data)
  freq = np.fft.fftfreq(len(fft_spectrum), 1/sample_rate)

  # Filter the frequency spectrum
  filtered_spectrum = fft_spectrum.copy()
  filtered_spectrum[np.where((freq < freq_range[0]) | (freq > freq_range[1]))] = 0

  # Inverse Fourier Transform to get the filtered audio
  filtered_audio = np.fft.ifft(filtered_spectrum)
  return filtered_audio.real

def visualize_frequency_change(audio_data, sample_rate, chunk_size=1024):
    # Divide audio data into chunks
    num_chunks = len(audio_data) // chunk_size
    audio_chunks = np.array_split(audio_data, num_chunks)

    # Process each chunk
    for idx,chunk in enumerate(audio_chunks):
        # Define frequency bands
        bass_freq_range = (20, 250)
        mid_freq_range = (250, 2000)
        treble_freq_range = (2000, sample_rate/2)

        # Apply band-pass filters
        bass_audio = filter_audio(chunk, sample_rate, bass_freq_range)
        mid_audio = filter_audio(chunk, sample_rate, mid_freq_range)
        treble_audio = filter_audio(chunk, sample_rate, treble_freq_range)

        # Visualize each band's frequency change
        time = np.arange(len(chunk)) / sample_rate  # Create time axis

        for audio, band_name in zip([bass_audio, mid_audio, treble_audio], ['Bass', 'Mid', 'Treble']):
            # Perform short-time Fourier Transform (STFT)
            spectrogram = librosa.stft(audio)
            # spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

            # Extract magnitude at each frequency bin
            frequencies = np.abs(spectrogram)

            # Plot the magnitude over time for each frequency bin
            plt.figure(figsize=(10, 4))
            for i in range(frequencies.shape[0]):
                plt.plot(time, frequencies[i])
                plt.xlabel('Time (s)')
                plt.ylabel('Magnitude')
                plt.title(f'{band_name} Frequency Change from sample {idx*chunk_size+1}')
                plt.tight_layout()
                st.pyplot(plt)

def visualize_frequency_distribution(audio_data, sample_rate):
    # Perform Fourier Transform
    fft_spectrum = np.fft.fft(audio_data)
    freq = np.fft.fftfreq(len(fft_spectrum), 1/sample_rate)

    # Define frequency bands
    bass_freq_range = (20, 250)
    mid_freq_range = (250, 2000)
    treble_freq_range = (2000, 20000)

    # Calculate energy in each band
    bass_energy = np.sum(np.abs(fft_spectrum[np.where((freq >= bass_freq_range[0]) & (freq <= bass_freq_range[1]))]))**2
    mid_energy = np.sum(np.abs(fft_spectrum[np.where((freq >= mid_freq_range[0]) & (freq <= mid_freq_range[1]))]))**2
    treble_energy = np.sum(np.abs(fft_spectrum[np.where((freq >= treble_freq_range[0]) & (freq <= treble_freq_range[1]))]))**2

    # Normalize energy values
    total_energy = bass_energy + mid_energy + treble_energy
    bass_percentage = (bass_energy / total_energy) * 100
    mid_percentage = (mid_energy / total_energy) * 100
    treble_percentage = (treble_energy / total_energy) * 100

    fig = px.bar(x=['Bass', 'Mid', 'Treble'], y=[bass_percentage, mid_percentage, treble_percentage],
                labels={'x': 'Frequency Band', 'y': 'Percentage of Energy'},
                title='Frequency Distribution')

    st.plotly_chart(fig)