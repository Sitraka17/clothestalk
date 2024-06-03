import streamlit as st
from PIL import Image
import numpy as np

# Trying to import TensorFlow, fallback to PyTorch if it fails
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
    tf_installed = True
except ModuleNotFoundError:
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    tf_installed = False

# Function to classify image using TensorFlow
def classify_image_tf(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Function to classify image using PyTorch
def classify_image_torch(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
    _, indices = torch.sort(outputs, descending=True)
    percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    classes = [(idx.item(), percentage[idx].item()) for idx in indices[0][:3]]
    return classes

# Load pre-trained model
if tf_installed:
    model = MobileNetV2(weights='imagenet')
else:
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    import requests
    labels = requests.get(LABELS_URL).json()

# Streamlit app
st.title("Clothing Classification and Brand Suggestion App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    if tf_installed:
        predictions = classify_image_tf(image)
        for pred in predictions:
            st.write(f"Predicted: {pred[1]} with confidence {pred[2]*100:.2f}%")
            if "shirt" in pred[1].lower():
                st.write("You might like shirts from Hugo Boss or Charles Tyrwhitt UK.")
            elif "scarf" in pred[1].lower():
                st.write("You might like scarves from Hermes.")
    else:
        predictions = classify_image_torch(image)
        for idx, confidence in predictions:
            st.write(f"Predicted: {labels[idx]} with confidence {confidence:.2f}%")
            if "shirt" in labels[idx].lower():
                st.write("You might like shirts from Hugo Boss or Charles Tyrwhitt UK.")
            elif "scarf" in labels[idx].lower():
                st.write("You might like scarves from Hermes.")

    st.write("If no brand was detected, we suggest the following brands:")
    st.write("For shirts: Hugo Boss, Charles Tyrwhitt UK")
    st.write("For scarves: Hermes")
