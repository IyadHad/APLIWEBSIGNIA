from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Initialize your model here
try:
    model = tf.keras.models.load_model('9mots500.h5')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")

actions = np.array(['Bonjour', 'Bravo', 'Ca va', 'Non', 'Oui', 'Au revoir', 'Pardon', 'Stp', 'Bien', 'Pas bien'])
sequence = []
sentence = []
threshold = 0.8

def mediapipe_detection(image, model):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image.flags.writeable = False  # Image is not writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
        return image, results
    except Exception as e:
        logging.error(f"Error in mediapipe_detection: {e}")
        return image, None

def draw_styled_landmarks(image, results):
    try:
        # Draw face landmarks
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
        # Draw left hand landmarks
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
        # Draw right hand landmarks
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
    except Exception as e:
        logging.error(f"Error in draw_styled_landmarks: {e}")

def extract_keypoints(results):
    try:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])
    except Exception as e:
        logging.error(f"Error in extract_keypoints: {e}")
        return np.zeros(33*4 + 468*3 + 21*3 + 21*3)

def gen_frames():  # generate frame by frame from camera
    global sequence, sentence  # Use the global variables

    try:
        cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                else:
                    image, results = mediapipe_detection(frame, holistic)
                    if results is None:
                        continue

                    draw_styled_landmarks(image, results)

                    # Prediction logic
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        # Predict here
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        logging.info(f"Prediction: {actions[np.argmax(res)]}")

                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                   
