# main_app.py

import streamlit as st
import numpy as np
import soundfile as sf
import io
from fir import apply_filters  # Import the apply_filters function

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

# Define a function to reset sliders
def reset_sliders():
    st.session_state.bass = 50
    st.session_state.mid = 50
    st.session_state.treble = 50
    st.session_state.reset_iteration += 1
    st.rerun()  # Rerun to update slider values immediately

# Initialize audio_data to None
audio_data = None
sample_rate = 44100  # Default sample rate

# Create tabs for current and real-time voice processing
tab1, tab2 = st.tabs(["Current", "Real-Time Voice Processing"])

with tab1:
    col0, col1, col2 = st.columns([5, 3, 2])

    with col0:
        col0.subheader("Audio Source")
        source = st.selectbox("Select Audio Source", ["Microphone", "File Upload"], placeholder="Select an audio source")

        if source == "File Upload":
            uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
            if uploaded_file:
                # Read the uploaded audio file
                audio_data, sample_rate = sf.read(uploaded_file)
                st.audio(uploaded_file, format="audio/wav")
        else:
            audio = st.experimental_audio_input("Record Audio", key="audio_input")
            if audio:
                audio_data = np.frombuffer(audio.read(), dtype=np.float32)
                st.download_button(label="Download the recorded audio", file_name="recorded_audio.wav", data=audio.read(),
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
            reset_sliders()  # Keep your reset function as is

with tab2:
    st.subheader("Real-Time Voice Processing")

    audio = st.experimental_audio_input("Record Audio", key="realtime_audio_input")
    if audio:
        audio_data = np.frombuffer(audio.read(), dtype=np.float32)

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
        st.success("Real-Time Voice Processing applied!")
    else:
        st.warning("Please record audio to apply real-time processing.")
