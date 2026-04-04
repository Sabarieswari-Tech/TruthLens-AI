import librosa
import numpy as np
import joblib

model = joblib.load("trained_models/audio_model.pkl")


def detect_fake_audio(audio_path):

    audio, sr = librosa.load(audio_path, duration=3)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    feature = np.mean(mfcc.T, axis=0)

    feature = feature.reshape(1, -1)

    prediction = model.predict(feature)[0]

    confidence = model.predict_proba(feature)[0].max()

    if prediction == 1:
        label = "Real Audio"
    else:
        label = "Fake Audio"

    return label, round(confidence * 100, 2)