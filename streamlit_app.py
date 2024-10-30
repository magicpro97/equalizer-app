import streamlit as st

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
    st.rerun()

# Layout columns for equalizer controls and actions
col0, col1, col2 = st.columns([5, 3, 2])

with col0:
    col0.subheader("Audio Source")
    source = st.selectbox("Select Audio Source", ["Microphone", "File Upload"], placeholder="Select an audio source")
    if source == "File Upload":
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
        if uploaded_file:
            st.audio(uploaded_file.read(), format="audio/wav")
    else:
        audio = st.experimental_audio_input("Upload an audio file", key="au dio_input")
        if audio:
            st.download_button(label="Download the recorded audio", file_name="recorded_audio.wav", data=audio.read(), mime="audio/wav")

with col1:
    col1.subheader("Equalizer Settings")
    st.session_state.bass = st.slider("Bass", min_value=0, max_value=100, value=st.session_state.bass,
                                      key=f"bass_slider_{st.session_state.reset_iteration}")
    st.session_state.mid = st.slider("Mid", min_value=0, max_value=100, value=st.session_state.mid,
                                     key=f"mid_slider_{st.session_state.reset_iteration}")
    st.session_state.treble = st.slider("Treble", min_value=0, max_value=100, value=st.session_state.treble,
                                        key=f"treble_slider_{st.session_state.reset_iteration}")

with col2:
    col2.subheader("Actions")
    if st.button("Apply Equalizer"):
        st.success("Equalizer settings applied!")
    if st.button("Reset Equalizer"):
        reset_sliders()