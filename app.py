import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# Load the pre-trained model
MODEL_PATH = './vgg16_model.h5'
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Please place the model file in the app directory.")
    st.stop()

model = load_model(MODEL_PATH)

# Define class labels (update this based on your specific dataset)
class_labels = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# Function to preprocess input image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match VGG16 input shape
    image = img_to_array(image)       # Convert to array
    image = image / 255.0            # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app interface
st.title("Maize Classification")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Display results
    st.write("### Prediction:")
    st.write(f"Class: {class_labels[predicted_class]} (Confidence: {confidence:.2f})")

else:
    st.write("Please upload an image to classify.")
