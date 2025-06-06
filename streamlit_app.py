import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import load_model
import tempfile
import os
from pydub import AudioSegment  # for mp3 to wav conversion

# === LOAD MODEL ===
try:
    model = load_model("model.h5")  # Try loading HDF5 model
except Exception as e1:
    try:
        model = load_model("my_model.keras")  # Fallback to keras format
    except Exception as e2:
        st.error("Error loading model files. Please check your model files.")
        st.stop()

# Class labels - adjust to your model's classes
class_names = ['Speech', 'Environment', 'Song']

# === STREAMLIT UI ===
st.set_page_config(page_title="🎧 Audio Classifier", layout="centered")
st.title("🎧 Audio Classification Web App")
st.markdown("""
Upload an audio file to classify it as:
- 🗣️ **Speech**
- 🌳 **Environment**
- 🎶 **Song**
""")

# === AUDIO FILE UPLOADER ===
audio_file = st.file_uploader("📁 Upload your audio file (WAV/MP3)", type=["wav", "mp3"])

if audio_file is not None:
    # Create temp file path
    suffix = ".wav"
    if audio_file.type == "audio/mp3":
        suffix = ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    # Convert mp3 to wav for librosa if needed
    if suffix == ".mp3":
        wav_tmp_path = tmp_path.replace(".mp3", ".wav")
        sound = AudioSegment.from_mp3(tmp_path)
        sound.export(wav_tmp_path, format="wav")
        os.remove(tmp_path)  # remove original mp3 temp file
        tmp_path = wav_tmp_path

    # Audio preview
    st.subheader("🔊 Audio Preview")
    st.audio(tmp_path)

    # Load audio - limit duration to 5 seconds for consistency
    y, sr = librosa.load(tmp_path, duration=5.0, offset=0.6)

    # Waveform plot
    st.subheader("📉 Waveform")
    fig1, ax1 = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title("Waveform")
    st.pyplot(fig1)

    # Mel spectrogram plot
    st.subheader("🔥 Mel Spectrogram")
    fig2, ax2 = plt.subplots()
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.f dB")
    ax2.set_title("Mel-frequency Spectrogram")
    st.pyplot(fig2)

    # Feature extraction (MFCC)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0).reshape(1, -1)

    # Model prediction with error handling
    st.subheader("🧠 Model Prediction")
    try:
        prediction = model.predict(mfccs_scaled)
        predicted_class = class_names[np.argmax(prediction)]
        st.success(f"🎶 Predicted class: **{predicted_class}**")

        # Confidence display
        st.subheader("📊 Prediction Confidence")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{class_names[i]}: {prob * 100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # Cleanup temp file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
