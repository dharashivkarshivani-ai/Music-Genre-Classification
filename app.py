import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import os

# ----------------------------
# Load your trained model
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("genre_classifier.keras")
    return model

model = load_model()

# ----------------------------
# Audio Processing Function
# ----------------------------
def extract_melspectrogram(file_path, sr=22050, n_mels=128):
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Fix shape to (128,128)
        if mel_db.shape[1] < 128:
            mel_db = np.pad(mel_db, ((0,0),(0,128 - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :128]
        mel_db = mel_db.reshape(1, 128, 128, 1)
        return mel_db
    except Exception as e:
        st.error(f"Failed to load audio: {e}")
        return None

# ----------------------------
# Genre Prediction
# ----------------------------
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

def predict_genre(file_path):
    mel_db = extract_melspectrogram(file_path)
    if mel_db is None:
        return None, None
    preds = model.predict(mel_db)
    genre_index = np.argmax(preds)
    confidence = float(np.max(preds))
    return GENRES[genre_index], confidence

# ----------------------------
# Streamlit UI
# ----------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Home Page
if app_mode == "Home":
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #181646;
        color: white;
    }
    h2, h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown("## Welcome to the Music Genre Classification System! 🎶🎧")
    st.image("music_genre_home.png", use_container_width=True)
    st.markdown("""
**Upload an audio file and the system will predict its genre.**
- Fast and accurate music classification.
- Supports .mp3 and .wav files.
""")

# About Project Page
elif app_mode == "About Project":
    st.header("About Project")
    st.markdown("""
Music experts and AI enthusiasts have been trying to classify audio into genres.
This project uses a deep learning model trained on Mel spectrograms extracted from audio files.
The dataset contains 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.
""")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Genre Prediction")
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if uploaded_file is not None:
        temp_path = os.path.join("temp_audio", uploaded_file.name)
        os.makedirs("temp_audio", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(temp_path, format="audio/mp3")

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                genre, confidence = predict_genre(temp_path)
                if genre:
                    st.success(f"Predicted Genre: {genre} ({confidence*100:.2f}% confidence)")
                else:
                    st.error("Failed to process the audio.")
