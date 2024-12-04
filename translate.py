import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import DATA_DIR
from datasets.labels_datasets import PETS_LABELS_DATASETS

# Step 1: Prepare dataset
labels_dict = PETS_LABELS_DATASETS

audio_files = list(labels_dict.keys())
labels = list(labels_dict.values())

# Step 2: Encode labels into integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


# Step 3: Function to preprocess audio files
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


# Step 4: Extract features for all files
features = []
for file_name in audio_files:
    file_path = f'{DATA_DIR}/{file_name}'
    feature = extract_features(file_path)
    if feature is not None:
        features.append(feature)

features = np.array(features)
encoded_labels = np.array(encoded_labels)

# Reshape features for CNN input
features = features[..., np.newaxis]  # Add channel dimension

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# Step 6: Build TensorFlow Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(features.shape[1], features.shape[2], 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Step 8: Save the Model
model.save("pet_translator_model.keras")
print("Model training complete and saved as 'pet_translator_model.keras'.")


# Step 9: Testing with new sample
def predict_audio(input_file_name):
    input_file_path = f'{DATA_DIR}/{input_file_name}'
    feature = extract_features(input_file_path)
    if feature is None:
        return "Error processing the file."
    feature = np.expand_dims(feature, axis=0)  # Add batch dimension
    feature = feature[..., np.newaxis]  # Add channel dimension
    prediction = model.predict(feature)
    pred_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return pred_label[0]


test_audio = 'dog_46.wav'  # Replace with your test file
print("Prediction:", predict_audio(test_audio))
