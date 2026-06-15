import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ================================================================
# SETTINGS
# ================================================================
DATASET_PATH = "data/genres"   # folder containing blues/, jazz/, rock/, etc.
SAMPLE_RATE = 22050
SEGMENT_DURATION = 3        # seconds
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION
N_MELS = 128
EXPECTED_MEL_SHAPE = (N_MELS, 128)

# ================================================================
# FEATURE EXTRACTION FOR 3-SECOND SEGMENTS
# ================================================================
def extract_segments(file_path, label):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    total_samples = len(y)
    segments = []
    segment_labels = []

    num_segments = total_samples // SAMPLES_PER_SEGMENT

    for i in range(num_segments):
        start = i * SAMPLES_PER_SEGMENT
        end = start + SAMPLES_PER_SEGMENT
        segment = y[start:end]

        mel = librosa.feature.melspectrogram(
            y=segment, sr=sr, n_mels=N_MELS, fmax=sr/2
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Resize/pad to 128×128
        if mel_db.shape[1] < 128:
            pad_width = 128 - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)))

        mel_db = mel_db[:, :128]

        segments.append(mel_db)
        segment_labels.append(label)

    return segments, segment_labels


# ================================================================
# LOAD FULL GTZAN — EACH FILE → 10 segments
# ================================================================
def load_dataset():
    X = []
    y = []

    genres = os.listdir(DATASET_PATH)

    for genre in genres:
        genre_path = os.path.join(DATASET_PATH, genre)

        if not os.path.isdir(genre_path):
            continue

        print(f"Processing Genre: {genre}")

        for file in os.listdir(genre_path):
            if file.endswith(".wav"):
                file_path = os.path.join(genre_path, file)
                segments, labels = extract_segments(file_path, genre)
                X.extend(segments)
                y.extend(labels)

    X = np.array(X)
    y = np.array(y)

    # Add channel dimension
    X = X.reshape(X.shape[0], N_MELS, 128, 1)

    return X, y


# ================================================================
# BUILD CNN MODEL
# ================================================================
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ================================================================
# MAIN TRAINING
# ================================================================
if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_dataset()

    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, shuffle=True, stratify=y_encoded
    )

    print("Building model...")
    model = build_model()

    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=25,
        batch_size=32
    )

    print("Evaluating...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nFinal Accuracy: {acc:.4f}")

    # SAVE
    model.save("genre_cnn_model.h5")
    np.save("label_classes.npy", le.classes_)

    print("Model saved successfully!")
