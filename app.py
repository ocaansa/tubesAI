from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

app = Flask(__name__)

# Load model dan label
model = load_model('model_lipreading.h5')
labels = np.load('labels.npy')

IMG_SIZE = 64

# Inisialisasi MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Webcam capture
cap = cv2.VideoCapture(0)

# 🔧 Tambahan variabel global
frame_count = 0
pred_label = ""

def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

def gen_frames():
    global frame_count, pred_label  
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1) 

        # Deteksi wajah dengan MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                h, w, _ = frame.shape
                xs = [lm.x * w for lm in face_landmarks.landmark]
                ys = [lm.y * h for lm in face_landmarks.landmark]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

                
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w), min(y2, h)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                frame_count += 1  

                if frame_count % 10 == 0: 
                    preprocessed = preprocess_frame(face_crop)
                    prediction = model.predict(preprocessed)
                    confidence = np.max(prediction)

                    if confidence > 1.0: 
                        pred_label = labels[np.argmax(prediction)]
                    else:
                        pred_label = ""

                
                cv2.putText(frame, f"Prediksi: {pred_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            pred_label = "Wajah tidak terdeteksi"  
            cv2.putText(frame, pred_label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
