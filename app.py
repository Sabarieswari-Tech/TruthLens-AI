# ------------------------------------------------------------
# TruthLens AI - Multimodal Fake Content Detection System
#
# This Flask application detects whether content is real or fake
# using machine learning and deep learning models.
#
# Supported Modules:
# 1. Text Fake News Detection
# 2. Image Manipulation Detection
# 3. Audio Authenticity Detection
# 4. Video Deepfake Detection
#
# Author: Sabarieswari S
# ------------------------------------------------------------

from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import tensorflow as tf
import librosa
import cv2

from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# ---------------- SUPPRESS TENSORFLOW WARNINGS ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------------- APP INITIALIZATION ----------------
app = Flask(__name__)

# ---------------- CONFIGURATION ----------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# ---------------- ALLOWED FILE TYPES ----------------
ALLOWED_IMAGE = {"png", "jpg", "jpeg"}
ALLOWED_AUDIO = {"wav", "mp3"}
ALLOWED_VIDEO = {"mp4", "avi", "mov"}

# ---------------- LOAD MACHINE LEARNING MODELS ----------------

# Text fake news model
text_model = joblib.load("trained_models/text_model.pkl")
vectorizer = joblib.load("trained_models/text_vectorizer.pkl")

# Image CNN model
image_model = tf.keras.models.load_model("trained_models/image_model.h5")

# Audio detection model
audio_model = joblib.load("trained_models/audio_model.pkl")

# -------- VIDEO MODEL (LOAD ONLY IF AVAILABLE) --------
video_model = None
video_model_path = "trained_models/video_model.h5"

if os.path.exists(video_model_path):
    video_model = tf.keras.models.load_model(video_model_path)
    print("✅ Video model loaded successfully")
else:
    print("⚠ Video model not found. Video detection disabled.")

# ---------------- FILE VALIDATION FUNCTION ----------------
def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        text_prediction=None,
        text_confidence=None,
        image_prediction=None,
        image_confidence=None,
        audio_prediction=None,
        audio_confidence=None,
        video_prediction=None,
        video_confidence=None
    )


# ---------------- ABOUT PAGE ----------------
@app.route("/about")
def about():
    return render_template("about.html")


# ---------------- DEVELOPER PAGE ----------------
@app.route("/developer")
def developer():
    return render_template("developer.html")


# ---------------- TEXT FAKE NEWS DETECTION ----------------
@app.route("/predict_text", methods=["POST"])
def predict_text():
    text = request.form.get("text")

    if not text:
        return render_template("index.html",
                               text_prediction="No text entered",
                               text_confidence=0)

    vector = vectorizer.transform([text])
    prediction = text_model.predict(vector)[0]

    score = text_model.decision_function(vector)[0]
    confidence = min(abs(score) * 50, 99.99)

    label = "Real News" if prediction == 1 else "Fake News"

    return render_template("index.html",
                           text_prediction=label,
                           text_confidence=round(confidence, 2))


# ---------------- IMAGE DETECTION ----------------
@app.route("/predict_image", methods=["POST"])
def predict_image():
    file = request.files.get("image")

    if not file or file.filename == "":
        return render_template("index.html",
                               image_prediction="No image selected")

    if not allowed_file(file.filename, ALLOWED_IMAGE):
        return render_template("index.html",
                               image_prediction="Invalid image format")

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    img = image.load_img(path, target_size=(128, 128))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = image_model.predict(img)[0][0]

    if prediction > 0.5:
        label = "Real Image"
        confidence = round(prediction * 100, 2)
    else:
        label = "Fake Image"
        confidence = round((1 - prediction) * 100, 2)

    return render_template("index.html",
                           image_prediction=label,
                           image_confidence=confidence)


# ---------------- AUDIO DETECTION ----------------
@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    file = request.files.get("audio")

    if not file or file.filename == "":
        return render_template("index.html",
                               audio_prediction="No audio selected")

    if not allowed_file(file.filename, ALLOWED_AUDIO):
        return render_template("index.html",
                               audio_prediction="Invalid audio format")

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    audio_data, sr = librosa.load(path, duration=3)

    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)

    prediction = audio_model.predict(features)[0]

    if hasattr(audio_model, "decision_function"):
        score = audio_model.decision_function(features)[0]
        confidence = min(abs(score) * 50, 99.99)
    else:
        confidence = round(audio_model.predict_proba(features)[0].max() * 100, 2)

    label = "Real Audio" if prediction == 1 else "Fake Audio"

    return render_template("index.html",
                           audio_prediction=label,
                           audio_confidence=round(confidence, 2))


# ---------------- VIDEO DEEPFAKE DETECTION ----------------
@app.route("/predict_video", methods=["POST"])
def predict_video():

    if video_model is None:
        return render_template("index.html",
                               video_prediction="Video model not available")

    file = request.files.get("video")

    if not file or file.filename == "":
        return render_template("index.html",
                               video_prediction="No video selected")

    if not allowed_file(file.filename, ALLOWED_VIDEO):
        return render_template("index.html",
                               video_prediction="Invalid video format")

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    cap = cv2.VideoCapture(path)

    predictions = []
    frame_count = 0
    MAX_FRAMES = 20

    while True:
        ret, frame = cap.read()

        if not ret or frame_count >= MAX_FRAMES:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        pred = video_model.predict(frame)[0][0]

        predictions.append(pred)
        frame_count += 1

    cap.release()

    if len(predictions) == 0:
        return render_template("index.html",
                               video_prediction="Video processing failed")

    avg_prediction = np.mean(predictions)

    if avg_prediction > 0.5:
        label = "Real Video"
        confidence = round(avg_prediction * 100, 2)
    else:
        label = "Fake Video"
        confidence = round((1 - avg_prediction) * 100, 2)

    return render_template("index.html",
                           video_prediction=label,
                           video_confidence=confidence)


# ---------------- RUN APPLICATION ----------------
if __name__ == "__main__":
    app.run(debug=True)