
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import load_model
import tempfile
import os

# === LOAD MODEL ===
try:
    # Try loading HDF5 format
    model = load_model("model.h5")
except:
    # Try loading native Keras format
    model = load_model("my_model.keras")

# Define class labels (adjust if your model has different classes)
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
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    # === AUDIO PREVIEW ===
    st.subheader("🔊 Audio Preview")
    st.audio(tmp_path)

    # === LOAD AUDIO ===
    y, sr = librosa.load(tmp_path, duration=5.0, offset=0.6)

    # === WAVEFORM PLOT ===
    st.subheader("📉 Waveform")
    fig1, ax1 = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title("Waveform")
    st.pyplot(fig1)

    # === MEL SPECTROGRAM ===
    st.subheader("🔥 Mel Spectrogram")
    fig2, ax2 = plt.subplots()
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.f dB")
    ax2.set_title("Mel-frequency Spectrogram")
    st.pyplot(fig2)

    # === FEATURE EXTRACTION ===
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0).reshape(1, -1)

    # === PREDICTION ===
    st.subheader("🧠 Model Prediction")
    prediction = model.predict(mfccs_scaled)
    predicted_class = class_names[np.argmax(prediction)]
    st.success(f"🎶 Predicted class: **{predicted_class}**")

    # === CONFIDENCE LEVELS ===
    st.subheader("📊 Prediction Confidence")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob * 100:.2f}%")

    # Cleanup
    os.remove(tmp_path)
