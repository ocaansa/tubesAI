import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATASET_PATH = 'dataset/WordAndPhrase'
IMG_SIZE = 64

X = []
labels = []

for label_folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label_folder)
    if not os.path.isdir(folder_path):
        continue

    for video_file in os.listdir(folder_path):
        if not video_file.endswith('.mp4'):
            continue

        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)

        success, frame = cap.read()
        frame_count = 0
        while success:
            if frame_count % 10 == 0:  # Ambil setiap 10 frame
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                X.append(frame)
                labels.append(label_folder)

            success, frame = cap.read()
            frame_count += 1

        cap.release()

print(f"Total data: {len(X)}")

X = np.array(X) / 255.0
labels = np.array(labels)

le = LabelEncoder()
y_enc = le.fit_transform(labels)
y_cat = tf.keras.utils.to_categorical(y_enc)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Simple CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save('model_lipreading.h5')
np.save('labels.npy', le.classes_)

print("Model trained and saved successfully.")
