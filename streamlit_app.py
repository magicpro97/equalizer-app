import streamlit as st
import numpy as np
import scipy.signal as signal
import soundfile as sf
import io

st.title("🎈 Equalizer")
st.write(
    "This is a simple equalizer app that allows you to adjust the levels of different frequency bands."
)

# Initialize session state for slider values and reset iteration
if 'bass' not in st.session_state:
    st.session_state.bass = 50
if 'mid' not in st.session_state:
    st.session_state.mid = 50
if 'treble' not in st.session_state:
    st.session_state.treble = 50
if 'reset_iteration' not in st.session_state:
    st.session_state.reset_iteration = 0

# Define filter frequency bands
SAMPLE_RATE = 44100  # Standard audio sample rate
bass_band = [20, 250]  # Bass frequencies
mid_band = [250, 4000]  # Mid frequencies
treble_band = [4000, 20000]  # Treble frequencies


# Function to create FIR filter for a given band and gain
def create_fir_filter(gain, band, numtaps=101, sample_rate=SAMPLE_RATE):
    nyquist = 0.5 * sample_rate
    band = [b / nyquist for b in band]  # Normalize band
    filter_coefs = signal.firwin(numtaps, band, pass_zero=False)
    return gain * filter_coefs


# Function to apply the filters based on slider values
def apply_filters(audio_data, bass_gain, mid_gain, treble_gain):
    # Create filters for each frequency band
    bass_filter = create_fir_filter(bass_gain, bass_band)
    mid_filter = create_fir_filter(mid_gain, mid_band)
    treble_filter = create_fir_filter(treble_gain, treble_band)

    # Check if audio_data is stereo or mono
    if audio_data.ndim == 1:  # Mono
        bass_filtered = signal.convolve(audio_data, bass_filter, mode="same")
        mid_filtered = signal.convolve(audio_data, mid_filter, mode="same")
        treble_filtered = signal.convolve(audio_data, treble_filter, mode="same")
        return bass_filtered + mid_filtered + treble_filtered
    elif audio_data.ndim == 2:  # Stereo
        # Apply filter to each channel separately
        bass_filtered = signal.convolve(audio_data[:, 0], bass_filter, mode="same")
        mid_filtered = signal.convolve(audio_data[:, 0], mid_filter, mode="same")
        treble_filtered = signal.convolve(audio_data[:, 0], treble_filter, mode="same")

        # Combine filtered channels
        filtered_audio = np.stack((bass_filtered + mid_filtered + treble_filtered,
                                   bass_filtered + mid_filtered + treble_filtered), axis=-1)
        return filtered_audio
    else:
        raise ValueError("Unsupported audio data dimensionality")


# Define a function to reset sliders
def reset_sliders():
    st.session_state.bass = 50
    st.session_state.mid = 50
    st.session_state.treble = 50
    st.session_state.reset_iteration += 1
    st.rerun()  # Rerun to update slider values immediately


# Layout columns for equalizer controls and actions
col0, col1, col2 = st.columns([5, 3, 2])

audio_data = None  # Initialize audio_data to None
sample_rate = SAMPLE_RATE  # Default sample rate

with col0:
    col0.subheader("Audio Source")
    source = st.selectbox("Select Audio Source", ["Microphone", "File Upload"], placeholder="Select an audio source")

    if source == "File Upload":
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
        if uploaded_file:
            # Read the uploaded audio file
            audio_data, sample_rate = sf.read(uploaded_file)
            st.audio(uploaded_file.read(), format="audio/wav")
    else:
        audio = st.experimental_audio_input("Record Audio", key="audio_input")
        if audio:
            audio_data = np.frombuffer(audio.read(), dtype=np.float32)
            st.download_button(label="Download the recorded audio", file_name="recorded_audio.wav", data=audio,
                               mime="audio/wav")

with col1:
    col1.subheader("Equalizer Settings")
    # Use unique keys for sliders to reset values
    st.session_state.bass = st.slider("Bass", min_value=0, max_value=100, value=st.session_state.bass,
                                      key=f"bass_slider_{st.session_state.reset_iteration}")
    st.session_state.mid = st.slider("Mid", min_value=0, max_value=100, value=st.session_state.mid,
                                     key=f"mid_slider_{st.session_state.reset_iteration}")
    st.session_state.treble = st.slider("Treble", min_value=0, max_value=100, value=st.session_state.treble,
                                        key=f"treble_slider_{st.session_state.reset_iteration}")

with col2:
    col2.subheader("Actions")
    if st.button("Apply Equalizer"):
        if audio_data is not None:
            # Apply FIR filter based on slider settings
            bass_gain = st.session_state.bass / 50.0  # Normalize to range 0.0 to 2.0
            mid_gain = st.session_state.mid / 50.0
            treble_gain = st.session_state.treble / 50.0

            # Process the audio with the filters
            filtered_audio = apply_filters(audio_data, bass_gain, mid_gain, treble_gain)

            # Save filtered audio to a buffer and play it
            filtered_audio_buffer = io.BytesIO()
            sf.write(filtered_audio_buffer, filtered_audio, sample_rate, format="wav")
            filtered_audio_buffer.seek(0)
            st.audio(filtered_audio_buffer, format="audio/wav")
            st.success("Equalizer settings applied!")
        else:
            st.warning("Please upload an audio file to apply the equalizer.")

    if st.button("Reset Equalizer"):
        reset_sliders()
