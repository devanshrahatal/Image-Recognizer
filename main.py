import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set logging level to suppress TensorFlow informational messages

import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore

# Load the trained model
model = load_model('image_recognizer.h5')

# Define the class names for CIFAR-10
obj_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def recognize_image():
    pic = input("\nInput Image Name (jpg / jpeg / png) : ")
    print("\nRecognizing...\n")
    
    # Load and store the original image
    original_img = cv.imread(pic)
    if original_img is None:
        print("Error: Unable to read image. Please check the file path and try again.")
        return

    # Convert the image to RGB format for displaying with matplotlib
    img = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)

    # Preprocess the image for the model
    resized_img = cv.resize(img, (32, 32), interpolation=cv.INTER_NEAREST)  # Resize without interpolation
    preprocessed_img = resized_img / 255.0

    # Predict the class of the image
    start_time = time.time()
    prediction = model.predict(np.array([preprocessed_img]))
    end_time = time.time()

    predicted_class_index = np.argmax(prediction)
    predicted_class_name = obj_names[predicted_class_index]
    prediction_time = end_time - start_time

    print(f"\nThe image is of : {predicted_class_name}")
    print(f"Prediction time : {prediction_time:.4f} seconds")

    disp = str(input("\nDo you want to see the image? (yes / no) : ")).lower()
    if disp == "yes":
        # Display the original image along with the predicted class name
        plt.figure(figsize=(8, 6))  # Adjust the size of the figure
        plt.imshow(img, aspect='auto')  # Adjust the image display properties
        plt.title(f"The image is of : {predicted_class_name}")
        plt.axis('off')
        plt.show()

print(f"\n", "IMAGE RECOGNIZER".center(50, "-"), "\n")

while True:
    recognize_image()
    another = str(input("\nDo you want to recognize another image? (yes / no) : ")).lower()
    if another != "yes":
        print(f"\n", "PROGRAM ENDED".center(50, "-"), "\n")
        break
