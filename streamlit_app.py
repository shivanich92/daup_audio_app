import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import train_test_split
import tempfile
import os
from pydub import AudioSegment

# =====================================
# ✅ STEP 1: TRAIN AND SAVE LSTM MODEL
# =====================================
# Simulate dummy MFCC data: 300 samples, 216 time steps, 40 features
X = np.random.rand(300, 216, 40)
y = np.random.randint(0, 3, size=(300,))
y_cat = to_categorical(y, num_classes=3)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(216, 40), return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 classes

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=16, validation_data=(X_test, y_test), verbose=0)

# Save model
model.save("model.h5")

# Class labels
class_names = ['Speech', 'Environment', 'Song']

# =====================================
# ✅ STEP 2: STREAMLIT APP BEGINS
# =====================================
st.set_page_config(page_title="🎧 Audio Classifier with LSTM", layout="centered")
st.title("🎧 Audio Classification with LSTM")
st.markdown("""
Upload an audio file to classify it as:
- 🗣️ **Speech**
- 🌳 **Environment**
- 🎶 **Song**
""")

# === LOAD MODEL ===
try:
    model = load_model("model.h5")
except Exception as e:
    st.error("❌ Model loading failed.")
    st.stop()

# === AUDIO FILE UPLOADER ===
audio_file = st.file_uploader("📁 Upload your audio file (WAV/MP3)", type=["wav", "mp3"])

if audio_file is not None:
    suffix = ".wav"
    if audio_file.type == "audio/mp3":
        suffix = ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    # Convert MP3 to WAV
    if suffix == ".mp3":
        wav_tmp_path = tmp_path.replace(".mp3", ".wav")
        sound = AudioSegment.from_mp3(tmp_path)
        sound.export(wav_tmp_path, format="wav")
        os.remove(tmp_path)
        tmp_path = wav_tmp_path

    st.subheader("🔊 Audio Preview")
    st.audio(tmp_path)

    # Load audio
    y, sr = librosa.load(tmp_path, duration=5.0, offset=0.6)

    # 📉 Waveform
    st.subheader("📉 Waveform")
    fig1, ax1 = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title("Waveform")
    st.pyplot(fig1)

    # 🔥 Mel Spectrogram
    st.subheader("🔥 Mel Spectrogram")
    fig2, ax2 = plt.subplots()
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.f dB")
    ax2.set_title("Mel-frequency Spectrogram")
    st.pyplot(fig2)

    # 🎛️ MFCCs (for LSTM)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = mfccs.T

    # Pad or truncate to 216 time steps
    desired_shape = 216
    if mfccs_scaled.shape[0] < desired_shape:
        pad_width = desired_shape - mfccs_scaled.shape[0]
        mfccs_scaled = np.pad(mfccs_scaled, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfccs_scaled = mfccs_scaled[:desired_shape, :]

    # Final shape for LSTM: (1, 216, 40)
    mfccs_scaled = mfccs_scaled.reshape(1, desired_shape, 40)

    # 🧠 Prediction
    st.subheader("🧠 Model Prediction")
    try:
        prediction = model.predict(mfccs_scaled)
        predicted_class = class_names[np.argmax(prediction)]
        st.success(f"🎶 Predicted class: **{predicted_class}**")

        # 📊 Confidence
        st.subheader("📊 Prediction Confidence")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{class_names[i]}: {prob * 100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # Cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
