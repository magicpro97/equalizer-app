import streamlit as st

st.title("🎈 Equalizer")
st.write(
    "This is a simple equalizer app that allows you to adjust the levels of different frequency bands."
)

# Add buttons to apply and reset the equalizer settings in the same row
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
