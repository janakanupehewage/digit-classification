Digit Classification App
This Digit Classification App uses a trained machine learning model to predict digits from uploaded images.

## Features
Upload Image: Users can upload an image of a digit (0-9).
Prediction: The app processes the image and predicts the digit using a Convolutional Neural Network (CNN) model.
Visualization: Displays prediction probabilities for each digit class.

## How to Use the App
Select 'Prediction' from the sidebar.
Upload an image of a digit (0-9) using the file uploader.
Click the 'Predict' button to see the classification result.

## Model Information
The model used is a Convolutional Neural Network (CNN) trained on the MNIST dataset.
It can classify digits from 0 to 9.
The input should be a grayscale image of size 28x28 pixels.

## Technologies used
Streamlit for the web application
TensorFlow/Keras for the machine learning model
Matplotlib and Seaborn for visualization

## Installation

To run the app locally, follow these steps:

1.Clone the repository:

	git clone <repository-url>  
	cd <repository-directory> 

2.Install the required packages:

	pip install streamlit tensorflow pillow matplotlib seaborn numpy

3.Run the app:

	streamlit run app.py

## Files in the Repository

app.py: The main Streamlit application script.
model/trained_model.h5: The pre-trained CNN model.
README.md: This file.

## Acknowledgements
The model is trained on the MNIST dataset, a large database of handwritten digits.
Thanks to the creators of Streamlit, TensorFlow/Keras, Matplotlib, and Seaborn for their amazing tools.