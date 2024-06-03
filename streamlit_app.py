import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load pre-trained model
model = MobileNetV2(weights='imagenet')

# Function to classify image
def classify_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Streamlit app
st.title("Clothing Classification and Brand Suggestion App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    predictions = classify_image(image)

    for pred in predictions:
        st.write(f"Predicted: {pred[1]} with confidence {pred[2]*100:.2f}%")

        # Brand suggestions based on predictions
        if "t-shirt" in pred[1].lower():
            st.write("You might like t-shirts from ASKET, Ash & Erie, H&M, Lululemon, or Carhartt.")
        elif "longsleeve" in pred[1].lower():
            st.write("You might like long sleeve shirts from Foret or Wax London.")
        elif "pants" in pred[1].lower():
            st.write("You might like pants from Levi's, Bonobos, or Lululemon.")
        elif "shoes" in pred[1].lower():
            st.write("You might like shoes from Nike, Adidas, or New Balance.")
        elif "shirt" in pred[1].lower():
            st.write("You might like shirts from Hugo Boss, Charles Tyrwhitt, Burberry, or Ralph Lauren.")
        elif "dress" in pred[1].lower():
            st.write("You might like dresses from Zara, Reformation, or Diane von Furstenberg.")
        elif "outwear" in pred[1].lower():
            st.write("You might like outerwear from The North Face, Patagonia, or Canada Goose.")
        elif "scarf" in pred[1].lower():
            st.write("You might like scarves from Hermes, Burberry, or Gucci.")
        # Add more clothing items and their respective brands as needed

    st.write("If no brand was detected, we suggest the following brands:")
    st.write("For t-shirts: ASKET, Ash & Erie, H&M, Lululemon, Carhartt")
    st.write("For long sleeve shirts: Foret, Wax London")
    st.write("For pants: Levi's, Bonobos, Lululemon")
    st.write("For shoes: Nike, Adidas, New Balance")
    st.write("For shirts: Hugo Boss, Charles Tyrwhitt, Burberry, Ralph Lauren")
    st.write("For dresses: Zara, Reformation, Diane von Furstenberg")
    st.write("For outerwear: The North Face, Patagonia, Canada Goose")
    st.write("For scarves: Hermes, Burberry, Gucci")
