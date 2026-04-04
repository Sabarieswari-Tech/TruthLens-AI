import cv2
import numpy as np
import tensorflow as tf

# load trained model
model = tf.keras.models.load_model("trained_models/video_model.h5")

# prediction function
def detect_fake_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while len(frames) < 20:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (224,224))

        frame = frame / 255.0

        frames.append(frame)

    cap.release()

    frames = np.array(frames)

    prediction = model.predict(frames)

    avg_prediction = np.mean(prediction)

    if avg_prediction > 0.5:
        return "Fake Video", avg_prediction
    else:
        return "Real Video", avg_prediction