import io
import queue
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import streamlit as st
from matplotlib import pyplot as plt
from pydub import AudioSegment

from fir import speed_up_audio, SAMPLE_RATE, apply_filters
from visualize import visualize_audio_file

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
bit_depth = 'PCM_24'  # Use 24-bit depth
DURATION = 10

def process_in_chunks(uploaded_file, chunk_size_ms=10000):
    audio_segment = AudioSegment.from_file(uploaded_file)
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

# Create tabs
tab1, tab2 = st.tabs(["Equalizer", "Real-time Filtering"])
with tab1:
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
                    audio_data = speed_up_audio(audio_data, 2.0)
            else:
                audio = st.audio_input("Record Audio", key="audio_input")
                if audio:
                    audio_data = np.frombuffer(audio.read(), dtype=np.int16).astype(np.float32)
                    audio_data /= np.iinfo(np.int16).max  # Normalize to [-1, 1]
                    audio_data = speed_up_audio(audio_data, 2.0)
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

            if st.button("Apply Equalizer"):
                if audio_data is not None:
                    # Apply FIR filter based on slider settings
                    bass_gain = st.session_state.bass / 50.0  # Normalize to range 0.0 to 2.0
                    mid_gain = st.session_state.mid / 50.0
                    treble_gain = st.session_state.treble / 50.0

                    with result_container:
                        filtered_audio, bass_filtered, mid_filtered, treble_filtered = visualize_audio_file(audio_data, visualize_container, bass_gain, mid_gain, treble_gain)
                        # Save filtered audio to a buffer and play it
                        filtered_audio_buffer = io.BytesIO()
                        sf.write(filtered_audio_buffer, filtered_audio, sample_rate, format="wav", subtype=bit_depth)
                        filtered_audio_buffer.seek(0)

                        st.audio(filtered_audio_buffer, format="audio/wav")
                        st.success("Equalizer settings applied!")
                else:
                    result_container.warning("Please upload an audio file to apply the equalizer.")

            if st.button("Reset Equalizer"):
                reset_sliders()  # Keep your reset function as is

# Global variables
current_frame = 0
should_stop = False

# Create a queue to hold the filtered audio data
audio_queue = queue.Queue()

with tab2:
    st.subheader("Real-time Audio Filtering")

    # Streamlit sliders for filter parameters
    bass_gain = st.slider('Bass Gain', 0.1, 10.0, 1.0, key='bass_gain_slider')
    mid_gain = st.slider('Mid Gain', 0.1, 10.0, 1.0, key='mid_gain_slider')
    treble_gain = st.slider('Treble Gain', 0.1, 10.0, 1.0, key='treble_gain_slider')

    # File uploader for local file
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"], key='file_uploader')

    # Placeholder for the waveform plot
    waveform_placeholder = st.empty()

    # Function to plot waveform
    def plot_waveform(filtered_chunk):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(filtered_chunk)
        ax.set_title("Real-time Audio Waveform")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        waveform_placeholder.pyplot(fig)
        plt.close(fig)

    # Stream the audio and apply the filter
    if uploaded_file and st.button('Play Audio with Real-time Filtering', key='play_button'):
        audio_segment = AudioSegment.from_file(uploaded_file)
        audio_data = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        audio_data /= np.iinfo(np.int16).max  # Normalize to [-1, 1]
        audio_data = speed_up_audio(audio_data, 2.0)

        def file_audio_callback(outdata, frames, time, status):
            global current_frame, should_stop
            if status:
                print(status)

            start = current_frame
            end = start + frames
            if start >= len(audio_data):
                should_stop = True
                return

            chunk = audio_data[start:end]
            if len(chunk) < frames:
                chunk = np.pad(chunk, (0, frames - len(chunk)), 'constant')

            filtered_chunk, _, _, _ = apply_filters(chunk, bass_gain, mid_gain, treble_gain)
            outdata[:] = np.column_stack((filtered_chunk, filtered_chunk))

            # Push the filtered chunk to the queue
            audio_queue.put(filtered_chunk)

            current_frame = end

        # Start the audio stream
        with sd.OutputStream(callback=file_audio_callback, samplerate=SAMPLE_RATE, channels=2, dtype='float32'):
            st.write("Playing audio with real-time filtering...")

            # Main loop to update the plot
            while current_frame < len(audio_data):
                if should_stop:
                    break

                # Check if there is new data in the queue
                try:
                    # Try to get data from the queue without blocking
                    filtered_chunk = audio_queue.get_nowait()
                    plot_waveform(filtered_chunk)
                except queue.Empty:
                    pass

                time.sleep(0.1)  # Adjust sleep time for smoother updates

    # Stop button to break the loop
    if st.button('Stop Audio', key='stop_button'):
        should_stop = True