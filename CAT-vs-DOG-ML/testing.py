import cv2
import os
import random
import numpy as np
import pickle as pick
import matplotlib.pyplot as plt
from keras.models import load_model

# Loading the trained model
model = load_model(
    '/home/rlns/Downloads/LANGUAGES/PYTHON/CAT vs DOG/trained_model.h5')

# Loading the preprocessed data
x = pick.load(open('/home/rlns/Downloads/LANGUAGES/PYTHON/CAT vs DOG/x.pkl',
              'rb')).astype('float32')/55.0

# Folder containing validation images
valid_folder = '/home/rlns/Downloads/LANGUAGES/PYTHON/CAT vs DOG/dogscats/valid/'

# List of categories
categories = ['cats', 'dogs']

# a list to store images and labels
images = []
labels = []

# Lists to store the indices of incorrect predictions
incorrect_indices = []

# Loop through categories and collect images
for category in categories:
    folder = os.path.join(valid_folder, category)
    image_files = os.listdir(folder)

    random.shuffle(image_files)
    # The first 10 images of each category
    for img_file in image_files[:8]:
        img_path = os.path.join(folder, img_file)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (175, 175))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        labels.append(category)

# Combine cat and dog images
combined_data = list(zip(images, labels))
random.shuffle(combined_data)
images, labels = zip(*combined_data)

# Convert to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# predictions for the selected images
predictions = model.predict(images)

# Interpret the predictions
class_names = ['Cat', 'Dog']
predicted_labels = [class_names[np.argmax(
    prediction)] for prediction in predictions]

# Displaying the images with their predicted labels
incorrect_count = 0
plt.figure(figsize=(12, 12))
for i in range(len(images)):
    plt.subplot(2, 8, i + 1)
    plt.imshow(images[i])
    actual_label = labels[i]
    predicted_label = predicted_labels[i]
    if actual_label == 'cats' and predicted_label == 'Cat' or actual_label == 'dogs' and predicted_label == 'Dog':
        plt.title(
            f'Predicted: {predicted_label}\nActual: {actual_label}', color='green')
    else:
        plt.title(
            f'Predicted: {predicted_label}\nActual: {actual_label}', color='red')
        incorrect_count += 1
    plt.axis('off')
plt.suptitle(f'Total Incorrect Predictions: {incorrect_count}', fontsize=20)
plt.show()
