import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Load model
model = load_model("potato_plant_model_v2.h5")

# Kelas (sesuaikan dengan dataset Anda)
CLASS_NAMES = ['Healthy', 'Late Blight', 'Early Blight']

# Konfigurasi halaman Streamlit
st.title("Potato Plant Disease Classification")
st.write("Upload an image of a potato leaf to identify its health status.")

# Upload gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess gambar
    st.write("Classifying...")
    img = image.resize((150, 150))  # Sesuaikan dengan ukuran input model
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    
    # Hasil prediksi
    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {np.max(predictions) * 100:.2f}%")
