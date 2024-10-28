import streamlit as st
from streamlit_mic_recorder import mic_recorder

st.title("🎈 Equalizer")
st.write(
    "This is a simple equalizer app that allows you to adjust the levels of different frequency bands."
)

recorder = None  # Placeholder for the recorder object
# Add buttons to apply and reset the equalizer settings in the same row
col0, col1, col2 = st.columns([5, 3, 2])

def audio_callback(audio_bytes):
    st.audio(audio_bytes, format="audio/wav")

with col0:
    col0.subheader("Audio Source")
    source = st.selectbox("Select Audio Source", ["Microphone", "File Upload"], placeholder="Select an audio source")

    col0_1, col0_2 = col0.columns([1, 5], vertical_alignment="center")
    if source == "File Upload":
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
        if uploaded_file:
            st.session_state.my_recorder_output = uploaded_file
            audio_callback(uploaded_file.read())
    else:
        with col0_1:
            recorder = mic_recorder(start_prompt="🔴", stop_prompt="⏹️", key='recorder')
        with col0_2:
            if recorder:
                audio_callback(recorder['bytes'])

with col1:
    col1.subheader("Equalizer Settings")
    st.write("Adjust the frequency bands:")
    bass = st.slider("Bass", min_value=0, max_value=100, value=50)
    mid = st.slider("Mid", min_value=0, max_value=100, value=50)
    treble = st.slider("Treble", min_value=0, max_value=100, value=50)

with col2:
    col2.subheader("Actions")
    if st.button("Apply Equalizer"):
        st.success("Equalizer settings applied!")
    if st.button("Reset Equalizer"):
        bass = 50
        mid = 50
        treble = 50
        st.rerun()  # Rerun the app to reset the sliders
