from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import tensorflow as tf
import librosa
import cv2

from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB upload limit

# ---------------- ALLOWED FILE TYPES ----------------
ALLOWED_IMAGE = {"png", "jpg", "jpeg"}
ALLOWED_AUDIO = {"wav", "mp3"}
ALLOWED_VIDEO = {"mp4", "avi", "mov"}

# ---------------- LOAD MODELS ----------------
text_model = joblib.load("trained_models/text_model.pkl")
vectorizer = joblib.load("trained_models/text_vectorizer.pkl")

image_model = tf.keras.models.load_model("trained_models/image_model.h5")
audio_model = joblib.load("trained_models/audio_model.pkl")
video_model = tf.keras.models.load_model("trained_models/video_model.h5")

# ---------------- FILE CHECK ----------------
def allowed_file(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed

# ---------------- HOME ----------------
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

# ---------------- TEXT DETECTION ----------------
@app.route("/predict_text", methods=["POST"])
def predict_text():
    text = request.form.get("text")
    if not text:
        return render_template("index.html",
                               text_prediction="No text entered",
                               text_confidence=0)

    vector = vectorizer.transform([text])
    pred = text_model.predict(vector)[0]

    # Use decision_function for confidence since LinearSVC doesn't have predict_proba
    score = text_model.decision_function(vector)[0]
    confidence = min(abs(score) * 50, 99.99)  # scale roughly to 0-100%

    label = "Real News" if pred == 1 else "Fake News"
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

    pred = image_model.predict(img)[0][0]

    if pred > 0.5:
        label = "Real Image"
        confidence = round(pred * 100, 2)
    else:
        label = "Fake Image"
        confidence = round((1 - pred) * 100, 2)

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

    pred = audio_model.predict(features)[0]
    # Linear model for audio: use decision_function if SVC-like
    if hasattr(audio_model, "decision_function"):
        score = audio_model.decision_function(features)[0]
        confidence = min(abs(score) * 50, 99.99)
    else:
        # fallback if model supports predict_proba
        confidence = round(audio_model.predict_proba(features)[0].max() * 100, 2)

    label = "Real Audio" if pred == 1 else "Fake Audio"
    return render_template("index.html",
                           audio_prediction=label,
                           audio_confidence=round(confidence, 2))

# ---------------- VIDEO DETECTION ----------------
@app.route("/predict_video", methods=["POST"])
def predict_video():
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

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > 20:
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

    avg_pred = np.mean(predictions)

    if avg_pred > 0.5:
        label = "Real Video"
        confidence = round(avg_pred * 100, 2)
    else:
        label = "Fake Video"
        confidence = round((1 - avg_pred) * 100, 2)

    return render_template("index.html",
                           video_prediction=label,
                           video_confidence=confidence)

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)