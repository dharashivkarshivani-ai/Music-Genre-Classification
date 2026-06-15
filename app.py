import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot as plt  # optional for visualization

# Function to cache model loading
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("Trained_model.keras")
    return model


# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds

    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    # Iterate over each chunk
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)

        # Resize using TensorFlow (no need to import tensorflow.image separately)
        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)


# TensorFlow Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])


# ---------------- HOME PAGE ----------------
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

    st.markdown('''## Welcome to the,\n## Music Genre Classification System! 🎶🎧''')
    image_path = "music_genre_home.png"
    st.image(image_path, use_container_width=True)

    st.markdown("""
**Our goal is to help identify music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**

### How It Works
1. **Upload Audio:** Go to the **Genre Classification** page and upload an audio file.
2. **Analysis:** Our system processes the audio using advanced algorithms.
3. **Results:** View the predicted genre with related information.

### Why Choose Us?
- **Accuracy:** Uses state-of-the-art deep learning models.
- **User-Friendly:** Simple and intuitive interface.
- **Fast:** Get results quickly and efficiently.

### Get Started
Click on the **Genre Classification** page in the sidebar to upload an audio file!

### About Us
Learn more about the project and our mission on the **About Project** page.
""")


# ---------------- ABOUT PAGE ----------------
elif app_mode == "About Project":
    st.markdown("""
    ### About Project
    Music experts have long tried to understand what differentiates one song from another.
    This project explores how AI can help us do that.

    ### About Dataset
    1. **genres original** - 10 genres, 100 audio files each (GTZAN dataset)
    2. **Genres:** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
    3. **images original** - Mel Spectrograms for each song for CNN-based classification
    4. **CSV files** - Contain extracted features for training
    """)


# ---------------- PREDICTION PAGE ----------------
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])

    if test_mp3 is not None:
        filepath = 'Test_Music/' + test_mp3.name
        with open(filepath, "wb") as f:
            f.write(test_mp3.getbuffer())

        if st.button("Play Audio"):
            st.audio(test_mp3)

        if st.button("Predict"):
            with st.spinner("Please Wait..."):
                X_test = load_and_preprocess_data(filepath)
                result_index = model_prediction(X_test)
                st.balloons()
                label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
                st.markdown(f"**🎵 Model Prediction:** It's a **:red[{label[result_index]}]** music!")
""