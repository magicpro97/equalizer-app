import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from fir import SAMPLE_RATE, apply_filters, apply_bandpass_filter


def visualize_audio_file(audio_bytes, container, bass_gain, mid_gain, treble_gain):
    print(f"Length original audio bytes: {len(audio_bytes)}")
    filtered_audio, bass_filtered, mid_filtered, treble_filtered = apply_filters(audio_bytes, bass_gain, mid_gain, treble_gain)
    print(f"Length filtered audio bytes: {len(filtered_audio)}")
    bass_original = apply_bandpass_filter(audio_bytes, 20, 250, 1.0, apply_gain=False)
    mid_original = apply_bandpass_filter(audio_bytes, 250, 4000, 1.0, apply_gain=False)
    treble_original = apply_bandpass_filter(audio_bytes, 4000, 20000, 1.0, apply_gain=False)
    visualize_bands(audio_bytes, filtered_audio, bass_filtered, mid_filtered, treble_filtered, bass_original, mid_original, treble_original, container)
    return filtered_audio, bass_filtered, mid_filtered, treble_filtered

def clean_audio_data(audio_data):
    return np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

def visualize_bands(origin_audio, filtered_audio, bass_filtered, mid_filtered, treble_filtered, bass_original, mid_original, treble_original, container):
    for audio, title in zip([origin_audio, filtered_audio, bass_original, bass_filtered, mid_original, mid_filtered, treble_original, treble_filtered],
                            ["Original Audio", "Filtered Audio", "Original Bass Band", "Filtered Bass Band", "Original Mid Band", "Filtered Mid Band", "Original Treble Band", "Filtered Treble Band"]):
        visualize_audio(audio, container, SAMPLE_RATE, title)

def visualize_audio(audio_data, container, sample_rate=SAMPLE_RATE, title="Audio Signal", n_fft=1024, hop_length=512):
    audio_data = clean_audio_data(audio_data)
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))  # Increase the height to add more space

    # Plot the waveform
    librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax[0])
    ax[0].set(title=f"Waveform - {title}", xlabel="Time (s)", ylabel="Amplitude")

    # Plot the spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    img = librosa.display.specshow(D, sr=sample_rate, x_axis="time", y_axis="log", hop_length=hop_length, ax=ax[1])
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    ax[1].set(title=f"Spectrogram - {title}", xlabel="Time (s)", ylabel="Frequency (Hz)")

    # Plot the frequency spectrum in dB
    freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
    fft_spectrum = np.abs(np.fft.rfft(audio_data))
    fft_spectrum_db = librosa.amplitude_to_db(fft_spectrum, ref=np.max)
    ax[2].plot(freqs, fft_spectrum_db)
    ax[2].set(title=f"Frequency Spectrum - {title}", xlabel="Frequency (Hz)", ylabel="Magnitude (dB)")

    plt.tight_layout(pad=3.0)  # Add padding between subplots
    container.pyplot(fig)
    plt.close(fig)