import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# load trained model
model = tf.keras.models.load_model("trained_models/image_model.h5")

# prediction function
def detect_fake_image(img_path):

    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return "Real Image", prediction
    else:
        return "Fake Image", prediction