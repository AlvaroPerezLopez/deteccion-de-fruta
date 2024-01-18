import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# URL del modelo comprimido en formato TensorFlow Lite
tflite_model_url = "https://github.com/AlvaroPerezLopez/deteccion-de-fruta/raw/main/compressed_model9_tf.tflite"

# Función para realizar la predicción con el modelo TensorFlow Lite
def predict_tflite(image_path, interpreter):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Obtener los detalles del input y output del modelo TensorFlow Lite
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Realizar la inferencia
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    return predictions

# Interfaz de usuario
st.title("97% Accuracy Fruit Detection App")

# Este código se ejecutará cada vez que haya un cambio en uploaded_file
uploaded_file = st.file_uploader("# Choose a fruit image (Apple, Banana, Grapes, Kiwi, Mango, Orange, Pineapple, Sugerapple or Watermelon)", type="jpg", key="fruit_image_upload")

if uploaded_file is not None:
    # Descargar el modelo TensorFlow Lite desde la URL
    interpreter = None
    with st.spinner("Loading TensorFlow Lite model..."):
        tflite_model_content = st.net.download_url_to_bytes(tflite_model_url).content
        interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
        interpreter.allocate_tensors()

    # Crear dos columnas
    col1, col2 = st.columns([2, 1])

    # Mostrar la imagen en la primera columna
    col1.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Hacer la predicción y mostrar "Classifying..." en la segunda columna
    with col2:
        # Mostrar "Classifying..." antes de la predicción
        st.markdown("## Classifying...")

        # Hacer la predicción con el modelo TensorFlow Lite
        predictions = predict_tflite(uploaded_file, interpreter)

        # Obtener el índice de la etiqueta predicha
        predicted_label_index = np.argmax(predictions)

        # Obtener el nombre de la etiqueta usando el índice
        predicted_label_name = label_names[predicted_label_index]

        # Mostrar la predicción con texto más grande usando Markdown
        st.markdown(f"## Prediction:\n\n## {predicted_label_name}")
