import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from config import DATA_DIR
from datasets.labels_datasets import LABELS_DATASETS

# Load the saved model
saved_model = tf.keras.models.load_model("pet_translator_model.keras", compile=False)

# Define the labels and encoder (same as in `translate.py`)
labels_dict = LABELS_DATASETS

labels = list(labels_dict.values())
label_encoder = LabelEncoder()
label_encoder.fit(labels)  # Reuse the same label encoder


# Function to preprocess audio files
def extract_features(file_path, n_mfcc=40, max_length=200):
    """
    Extract MFCC features from audio file.
    - n_mfcc: Number of MFCC coefficients
    - max_length: Padding/truncating length
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)  # Load audio
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)  # Extract MFCC features
        mfcc = np.pad(mfcc, ((0, 0), (0, max(0, max_length - mfcc.shape[1]))), mode='constant')  # Pad
        return mfcc[:, :max_length]  # Truncate to max_length
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Predict function
def predict_audio(input_file_name):
    input_file_path = f'{DATA_DIR}/{input_file_name}'
    feature = extract_features(input_file_path)
    if feature is None:
        return "Error processing the file."
    feature = np.expand_dims(feature, axis=0)  # Add batch dimension
    feature = feature[..., np.newaxis]  # Add channel dimension
    prediction = saved_model.predict(feature)
    pred_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return pred_label[0]


# Test with an audio file
test_audio = 'dog_1.wav'  # Replace with the actual test file name
print("Prediction:", predict_audio(test_audio))
