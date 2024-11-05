import numpy as np
import matplotlib.pyplot as plt
import librosa
import plotly.express as px
import streamlit as st
from scipy.signal import butter, lfilter
from fir import SAMPLE_RATE

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
def visualize_audio_file(file, container, n_fft=2048, hop_length=512):
    audio, sr = librosa.load(file)

    # STFT
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    stft_magnitude = np.abs(stft)

    # Get frequency bins
    freq = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Define frequency bands
    bass_freq_range = (20, 250)
    mid_freq_range = (250, 4000)
    treble_freq_range = (4000, sr//2)

    # Create masks for each frequency band
    bass_mask = (freq >= bass_freq_range[0]) & (freq <= bass_freq_range[1])
    mid_mask = (freq >= mid_freq_range[0]) & (freq <= mid_freq_range[1])
    treble_mask = (freq >= treble_freq_range[0]) & (freq <= treble_freq_range[1])

    # Apply masks to the STFT magnitude
    bass_stft = stft_magnitude.copy()
    bass_stft[~bass_mask] = 0

    mid_stft = stft_magnitude.copy()
    mid_stft[~mid_mask] = 0

    treble_stft = stft_magnitude.copy()
    treble_stft[~treble_mask] = 0

    # Convert back to time-domain signals
    bass_audio = librosa.istft(bass_stft, hop_length=512)
    mid_audio = librosa.istft(mid_stft, hop_length=512)
    treble_audio = librosa.istft(treble_stft, hop_length=512)

    container.markdown("# Visualize")
    # Visualize each band's spectrogram
    for band_audio, band_name in zip([bass_audio, mid_audio, treble_audio], ['Bass', 'Mid', 'Treble']):
        spectrogram = librosa.feature.melspectrogram(y=band_audio, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'{band_name} Frequency Spectrogram')
        plt.tight_layout()
        container.pyplot(plt)

def visualize_bands(bass_filtered, mid_filtered, treble_filtered, container):
    for band_audio, band_name in zip([bass_filtered, mid_filtered, treble_filtered], ['Bass', 'Mid', 'Treble']):
        spectrogram = librosa.feature.melspectrogram(y=band_audio, sr=SAMPLE_RATE)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram_db, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'{band_name} Frequency Spectrogram')
        plt.tight_layout()
        container.pyplot(plt)

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