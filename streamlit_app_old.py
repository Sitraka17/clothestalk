import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests

# Load pre-trained model (e.g., MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Mapping ImageNet class IDs to human-readable labels
imagenet_labels = np.array(open('imagenet_labels.txt').read().splitlines())

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Function to get predictions
def get_predictions(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    top_pred_idx = np.argmax(predictions)
    top_pred_label = imagenet_labels[top_pred_idx]
    return top_pred_label

# Function to search brand if not identified
def search_brand(clothing_item):
    url = f"https://api.example.com/search?q={clothing_item}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data and 'brand' in data:
            return data['brand']
    return "Brand not found"

# Extended list of known brands
known_brands = [
    "Nike", "Adidas", "Puma", "Reebok", "Under Armour", "Levi's", "Gap", "H&M", "Zara", "Uniqlo",
    "Gucci", "Prada", "Versace", "Louis Vuitton", "Chanel", "Hermes", "Dior", "Burberry", "Ralph Lauren",
    "Tommy Hilfiger", "Calvin Klein", "Michael Kors", "Hugo Boss", "Armani", "Dolce & Gabbana", "Fendi",
    "Balenciaga", "Saint Laurent", "Valentino", "Givenchy", "Kenzo", "Lacoste", "Supreme", "Off-White",
    "Stone Island", "Moncler", "Patagonia", "North Face", "Columbia", "Converse", "New Balance",
    "Vans", "Timberland", "Dr. Martens", "Brooks Brothers", "Abercrombie & Fitch", "Hollister", "Aeropostale",
    "Diesel", "Guess", "Bvlgari", "Tory Burch", "Kate Spade", "Marc Jacobs", "Coach", "Vera Wang", "Stella McCartney",
    "Alexander McQueen", "Vivienne Westwood", "Jean Paul Gaultier", "Karl Lagerfeld", "Moschino", "Miu Miu",
    "Salvatore Ferragamo", "Ermenegildo Zegna", "Brioni", "Canali", "Giorgio Armani", "Paul Smith", "Ted Baker",
    "Fred Perry", "J.Crew", "Banana Republic", "Lululemon", "Alo Yoga", "Gymshark", "Fila", "ASICS",
    "Marmot", "Spyder", "Helly Hansen", "Barbour", "Gore-Tex", "Arc'teryx", "Canada Goose", "Eddie Bauer",
    "LL Bean", "Carhartt", "Dickies", "Wrangler", "Lee", "Dockers", "Champion", "Russell Athletic", "Fila"
]

# Streamlit UI
st.title("Clothes Classifier and Brand Recognizer")

uploaded_file = st.file_uploader("Choose a PNG image", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    label = get_predictions(image)
    st.write(f"Identified clothing item: {label}")
    
    # Identify brand
    brand = None
    for known_brand in known_brands:
        if known_brand.lower() in label.lower():
            brand = known_bra
