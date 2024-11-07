# main_app.py
import io

import numpy as np
import soundfile as sf
import streamlit as st
from pydub import AudioSegment

from fir import apply_filters  # Import the apply_filters function
from visualize import visualize_bands #, visualize_audio_file

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
uploaded_file = None
audio_data = None
sample_rate = 44100  # Default sample rate

def process_in_chunks(uploaded_file, chunk_size_ms=10000):
    audio_segment = AudioSegment.from_file(uploaded_file, format="mp3")
    total_length_ms = len(audio_segment)
    chunk_size_ms = total_length_ms

    audio_data = None
    for start_ms in range(0, total_length_ms, chunk_size_ms):
        print(f"Processing chunk {start_ms} - {start_ms + chunk_size_ms} ms")
        end_ms = min(start_ms + chunk_size_ms, total_length_ms)
        chunk = audio_segment[start_ms:end_ms]

        # Process the chunk (convert to numpy, normalize, etc.)
        audio_data = np.array(chunk.get_array_of_samples())
        audio_data = audio_data.astype(np.float32)
        audio_data /= np.iinfo(np.int16).max  # Normalize to [-1, 1]

    return audio_data

main_container = st.container()

result_container = st.container()

visualize_container = st.container()

# Layout columns for equalizer controls and actions
with main_container:
    col0, col1, col2 = st.columns([5, 3, 2])

    with col0:
        col0.subheader("Audio Source")
        source = st.selectbox("Select Audio Source", ["Microphone", "File Upload"])

        if source == "File Upload":
            uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
            if uploaded_file:
                # Read the uploaded audio file
                st.audio(uploaded_file, format="audio/wav")
                audio_data = process_in_chunks(uploaded_file)
        else:
            audio = st.experimental_audio_input("Record Audio", key="audio_input")
            if audio:
                audio_data = np.frombuffer(audio.read(), dtype=np.float32)
                st.download_button(label="Download the recorded audio", file_name="recorded_audio.wav",
                                   data=audio.read(),
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

        # if st.button("Visualize"):
        #     bass_gain = st.session_state.bass / 50.0  # Normalize to range 0.0 to 2.0
        #     mid_gain = st.session_state.mid / 50.0
        #     treble_gain = st.session_state.treble / 50.0
        #     if uploaded_file is not None:
        #         visualize_audio_file(audio_data, container=visualize_container, bass_gain=bass_gain, mid_gain=mid_gain,
        #                             treble_gain=treble_gain)
        #     elif audio is not None:
        #         visualize_audio_file(audio_data, container=visualize_container, bass_gain=bass_gain, mid_gain=mid_gain,
        #                             treble_gain=treble_gain)
        #     else:
        #         result_container.warning("Please upload an audio file to visualize.")

        if st.button("Apply Equalizer"):
            if audio_data is not None:
                # Apply FIR filter based on slider settings
                bass_gain = st.session_state.bass / 50.0  # Normalize to range 0.0 to 2.0
                mid_gain = st.session_state.mid / 50.0
                treble_gain = st.session_state.treble / 50.0

                with result_container:
                    # Process the audio with the filters
                    filtered_audio, bass_filtered, mid_filtered, treble_filtered = apply_filters(audio_data, bass_gain,
                                                                                                 mid_gain, treble_gain)
                    # Save filtered audio to a buffer and play it
                    filtered_audio_buffer = io.BytesIO()
                    sf.write(filtered_audio_buffer, filtered_audio, sample_rate, format="wav")
                    filtered_audio_buffer.seek(0)

                    visualize_bands(audio_data, filtered_audio, bass_filtered, mid_filtered, treble_filtered,
                                    visualize_container)

                    # Make audio sorter than a half duration and speed faster x2
                    filtered_audio = AudioSegment.from_file(filtered_audio_buffer, format="wav")
                    filtered_audio = filtered_audio.speedup(playback_speed=2.0)
                    filtered_audio.export(filtered_audio_buffer, format="wav")

                    st.audio(filtered_audio_buffer, format="audio/wav")
                    st.success("Equalizer settings applied!")
            else:
                result_container.warning("Please upload an audio file to apply the equalizer.")

        if st.button("Reset Equalizer"):
            reset_sliders()  # Keep your reset function as is

# with visualize_container:
# st.markdown("# Visualize")
