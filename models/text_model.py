import joblib

# load model and vectorizer
model = joblib.load("trained_models/text_model.pkl")
vectorizer = joblib.load("trained_models/text_vectorizer.pkl")

def detect_fake_text(text):

    text_tfidf = vectorizer.transform([text])

    prediction = model.predict(text_tfidf)[0]

    # confidence score
    score = model.decision_function(text_tfidf)[0]

    confidence = abs(score) * 100

    if confidence > 100:
        confidence = 99.99

    if prediction == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return result, round(confidence, 2)