import cv2
import mediapipe as mp
import numpy as np
import pickle
import json
from flask import Flask, render_template, Response, jsonify
import os

app = Flask(__name__)

# Load model
MODEL_PATH = './'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model_dict = pickle.load(open(MODEL_PATH, 'rb'))
model = model_dict.get('model')

if model is None:
    raise ValueError("Model not found in the loaded model dictionary.")

# Load labels dynamically from labels.json
LABELS_PATH = ''
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")

with open(LABELS_PATH, 'r') as f:
    labels_dict = json.load(f)

# Convert keys to integers
labels_dict = {int(k): v for k, v in labels_dict.items()}

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

predicted_character = "No sign detected"

# Video feed generator
def generate_video():
    global predicted_character
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                data_aux = []
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                min_x, min_y = min(x_coords), min(y_coords)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                # Define bounding box with padding
                x1 = max(int(min(x_coords) * W) - 20, 0)
                y1 = max(int(min(y_coords) * H) - 20, 0)
                x2 = min(int(max(x_coords) * W) + 20, W)
                y2 = min(int(max(y_coords) * H) + 20, H)

                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    label_key = int(prediction[0])
                    predicted_character = labels_dict.get(label_key, "Unknown Sign")
                except Exception as e:
                    print(f"Prediction error: {e}")
                    predicted_character = "Error"

                # Add rectangle and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(
                    frame,
                    predicted_character,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA
                )

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
cap.release()
cv2.destroyAllWindows()
