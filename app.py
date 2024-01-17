import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
 
# # URL del modelo en GitHub
# model_url = 'https://github.com/AlvaroPerezLopez/deteccion-de-fruta/raw/main/model9.keras'

# # Función para cargar el modelo9
# @st.cache_data()
# def load_model():
#     # Comprobar si el modelo ya está descargado
#     if not os.path.exists('model9.keras'):
#         # Descargar el modelo desde GitHub
#         response = requests.get(model_url)
#         with open('model9.keras', 'wb') as f:
#             f.write(response.content)

#     return tf.keras.models.load_model('model9.keras')

# # Llamada a la función para cargar el modelo9
# model9 = load_model()

# # Función para realizar la predicción
# def predict(image_path, model):
#     img = Image.open(image_path)
#     img = img.resize((244, 244))
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
#     predictions = model.predict(img_array)
#     return predictions

# # Interfaz de usuario
# st.title("Fruit Detection App")

# # Este código solo se ejecutará cuando la aplicación no se esté construyendo en Netlify
# uploaded_file = st.file_uploader("Choose a fruit image...", type="jpg")

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
#     st.write("")
#     st.write("Classifying...")

#     # Hacer la predicción
#     predictions = predict(uploaded_file, model9)

#     st.write("Prediction:")
#     st.write(predictions)

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
