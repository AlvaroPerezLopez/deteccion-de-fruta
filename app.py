# app.py
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

# Cargar el modelo
model = tf.keras.models.load_model('model9.keras')

st.title('Detención de Frutas')

# Interfaz para subir una imagen
uploaded_file = st.file_uploader("Elige una imagen de fruta...", type="jpg")

if uploaded_file is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, caption='Imagen subida', use_column_width=True)

    # Preprocesar la imagen para realizar la predicción
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Realizar la predicción
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    st.write("\n\nPredicciones:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i + 1}: {label} ({score:.2f})")
