# TruthLens AI

TruthLens AI is a multimodal fake content detection system that uses Artificial Intelligence and Machine Learning to determine whether digital content is real or fake.
The system analyzes **text, images, audio, and video** to identify misinformation and manipulated media.

The goal of this project is to help combat the spread of fake news and deepfakes by providing an automated verification tool.

---

## Features

* Fake news detection from text content
* Image manipulation detection
* Audio authenticity analysis
* Video deepfake detection
* Web-based interface for easy interaction
* Machine learning and deep learning models for prediction

---

## Technologies Used

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python
* Flask

### Machine Learning

* TensorFlow
* scikit-learn
* joblib

### Media Processing

* OpenCV
* librosa

---

## System Workflow

Input Content (Text / Image / Audio / Video)

↓

Preprocessing

↓

Feature Extraction

↓

Machine Learning / Deep Learning Model

↓

Prediction Output (Real or Fake)

---

## Models Used

### Text Detection

* TF-IDF Vectorization
* Support Vector Machine (SVM)

### Image Detection

* Convolutional Neural Network (CNN)

### Audio Detection

* Audio feature extraction using Librosa
* Deep learning classifier

### Video Detection

* Frame extraction using OpenCV
* CNN-based deepfake detection

---

## Project Structure

TruthLens-AI

app.py

requirements.txt

README.md

models/

trained_models/

static/

templates/

screenshots/

training_scripts/

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sabarieswari-Tech/TruthLens-AI.git
cd TruthLens-AI
```

### 2. Install required dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

### 4. Open the web application

Open your browser and go to:

```
http://127.0.0.1:5000
```

---

## Screenshots

### Model Training Output

Add your training screenshot here.

Example:

![Training Output](screenshots/training_output.png)

### Web Interface

![Interface](screenshots/interface.png)

---

## Note

Due to GitHub file size limitations, trained models and datasets are not included in this repository.

---

## Future Improvements

* Improve video deepfake detection accuracy
* Deploy the application to cloud platforms
* Add real-time misinformation detection
* Support multilingual content analysis
* Improve model performance with larger datasets

---

## Author

Sabarieswari S

GitHub: https://github.com/Sabarieswari-Tech

---

## License

This project is licensed under the MIT License.
