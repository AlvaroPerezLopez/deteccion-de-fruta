import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)

# Cargar el modelo9 en el momento adecuado
model9_path = 'model9.keras'

# Función para cargar el modelo9
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model(model9_path)

# Llamada a la función para cargar el modelo9
model9 = load_model()

# Función para realizar la predicción
def predict(image_path, model):
    img = Image.open(image_path)
    img = img.resize((244, 244))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    return predictions

# Interfaz de usuario
st.title("Fruit Detection App")

uploaded_file = st.file_uploader("Choose a fruit image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Hacer la predicción
    predictions = predict(uploaded_file, model9)

    st.write("Prediction:")
    st.write(predictions)
