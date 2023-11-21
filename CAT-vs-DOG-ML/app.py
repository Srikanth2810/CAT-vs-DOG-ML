from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
import pickle as pick
from keras.models import load_model
import matplotlib.pyplot as plt
import signal
import sys
import webbrowser  # Import the webbrowser module

app = Flask(__name__)

# Load the trained model
model = load_model(
    '/home/rlns/Downloads/LANGUAGES/PYTHON/CAT vs DOG/trained_model.h5')

# Load the preprocessed data (x.pkl)
x = pick.load(open('/home/rlns/Downloads/LANGUAGES/PYTHON/CAT vs DOG/x.pkl',
              'rb')).astype('float32') / 55.0

# Directory to store uploaded images temporarily
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def predict_and_display(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (175, 175))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 55.0
    prediction = model.predict(image)
    class_names = ['Cat', 'Dog']
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class, image


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload and prediction here
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded image temporarily
            image_path = os.path.join(
                app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(image_path)

            # Predict the selected image
            predicted_class, image = predict_and_display(image_path)

            # Provide a URL to the temporary image for rendering in the template
            image_url = url_for(
                'uploaded_file', filename=uploaded_file.filename)

            return render_template('result.html', predicted_class=predicted_class, image_url=image_url)
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Serve the uploaded image
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Signal handler for shutdown


def handle_shutdown(signal, frame):
    print("Shutting down...")
    # Add any cleanup code here if needed.
    sys.exit(0)


# Register signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handle_shutdown)

webbrowser.open('http://127.0.0.1:5000')

app.run(debug=True)
