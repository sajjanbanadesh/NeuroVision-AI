from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# ====================================
# LOAD TRAINED MODEL
# ====================================

model = tf.keras.models.load_model(
    "alzheimer_model.h5"
)

# ====================================
# CLASS NAMES
# ====================================

classes = [
    'MildDemented',
    'ModerateDemented',
    'NonDemented',
    'VeryMildDemented'
]

# ====================================
# UPLOAD FOLDER
# ====================================

UPLOAD_FOLDER = 'uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ====================================
# HOME PAGE
# ====================================

@app.route('/')
def home():

    return render_template('index.html')

# ====================================
# PREDICTION ROUTE
# ====================================

@app.route('/predict', methods=['POST'])
def predict():

    # Check file upload
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # ====================================
    # SAVE IMAGE
    # ====================================

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    # ====================================
    # LOAD IMAGE
    # ====================================

    img = Image.open(filepath).convert('RGB')

    img = img.resize((128,128))

    # Convert image to array
    img_array = np.array(img)

    # IMPORTANT:
    # DO NOT normalize here
    # because model already has:
    # layers.Rescaling(1./255)

    # Expand dimensions
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ====================================
    # PREDICTION
    # ====================================

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = classes[predicted_index]

    confidence = float(
        np.max(prediction) * 100
    )

    # ====================================
    # CLASS PROBABILITIES
    # ====================================

    probabilities = {

        classes[i]:
        round(float(prediction[0][i]) * 100, 2)

        for i in range(len(classes))
    }

    # ====================================
    # RENDER RESULT
    # ====================================

    return render_template(

        'index.html',

        prediction=predicted_class,

        confidence=round(confidence, 2),

        probabilities=probabilities,

        image_path=filepath
    )

# ====================================
# RUN APP
# ====================================

if __name__ == '__main__':

    app.run(debug=True)