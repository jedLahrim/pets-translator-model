import os
import tempfile

import librosa
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from sklearn.preprocessing import LabelEncoder

from datasets.labels_datasets import PETS_LABELS_DATASETS

app = Flask(__name__)
CORS(app)

# Load the trained model
saved_model = tf.keras.models.load_model("pet_translator_model.keras", compile=False)

# Define the labels and encoder (same as in your translation.py)
labels_dict = PETS_LABELS_DATASETS
labels = list(labels_dict.values())
label_encoder = LabelEncoder()
label_encoder.fit(labels)


def extract_features(file_path, n_mfcc=40, max_length=200):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc = np.pad(mfcc, ((0, 0), (0, max(0, max_length - mfcc.shape[1]))), mode='constant')
        return mfcc[:, :max_length]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


@app.route('/', methods=['GET'])
def welcome():
    return jsonify({'message':'hello from the server'}),200


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            file_path = temp_file.name

        # Extract features from the audio
        feature = extract_features(file_path)

        # Clean up the temporary file
        os.unlink(file_path)

        if feature is None:
            return jsonify({'error': 'Error processing the file'}), 400

        feature = np.expand_dims(feature, axis=0)  # Add batch dimension
        feature = feature[..., np.newaxis]  # Add channel dimension
        prediction = saved_model.predict(feature)

        # Get the predicted label
        pred_label = label_encoder.inverse_transform([np.argmax(prediction)])
        return jsonify({'label': pred_label[0]})

    except Exception as e:
        return jsonify({'error': f"Error during prediction: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=False, host="localhost", port=5002)
