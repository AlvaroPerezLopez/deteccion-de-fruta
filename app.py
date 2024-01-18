import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Cargar el modelo TensorFlow Lite desde el archivo local
tflite_model_path = "https://github.com/AlvaroPerezLopez/deteccion-de-fruta/blob/main/compressed_model9_tf.tflite"
tflite_model = tf.lite.Interpreter(model_path=tflite_model_path)
tflite_model.allocate_tensors()

# Lista de nombres de etiquetas en el orden correcto
label_names = ["Apple", "Banana", "Grapes", "Kiwi", "Mango", "Orange", "Pineapple", "Sugerapple", "Watermelon"]

# Función para realizar la predicción
def predict(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Ingresar la imagen al modelo TensorFlow Lite
    input_tensor_index = tflite_model.get_input_details()[0]['index']
    output = tflite_model.tensor(tflite_model.get_output_details()[0]['index'])

    tflite_model.set_tensor(input_tensor_index, img_array)
    tflite_model.invoke()

    # Obtener las predicciones
    predictions = output()[0]
    return predictions

# Interfaz de usuario
st.title("97% Accuracy Fruit Detection App")

# Este código se ejecutará cada vez que haya un cambio en uploaded_file
uploaded_file = st.file_uploader("# Choose a fruit image (Apple, Banana, Grapes, Kiwi, Mango, Orange, Pineapple, Sugerapple or Watermelon)", type="jpg", key="fruit_image_upload")

if uploaded_file is not None:
    # Crear dos columnas
    col1, col2 = st.columns([2, 1])

    # Mostrar la imagen en la primera columna
    col1.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Hacer la predicción y mostrar "Classifying..." en la segunda columna
    with col2:

        # Mostrar "Classifying..." antes de la predicción
        st.markdown("## Classifying...")

        # Hacer la predicción
        predictions = predict(uploaded_file)

        # Obtener el índice de la etiqueta predicha
        predicted_label_index = np.argmax(predictions)

        # Obtener el nombre de la etiqueta usando el índice
        predicted_label_name = label_names[predicted_label_index]

        # Mostrar la predicción con texto más grande usando Markdown
        st.markdown(f"## Prediction:\n\n## {predicted_label_name}")
