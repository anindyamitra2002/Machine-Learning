import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
from mtcnn import MTCNN


model = tf.keras.models.load_model('custom_model_bw.h5')
st.set_page_config(page_title='AGENDER predictor')
# Load the MTCNN model
mtcnn_model = MTCNN()


def expand_bounding_box(x, y, width, height, expansion):
    expanded_x = x - expansion
    expanded_y = y - expansion
    expanded_width = width + 2 * expansion
    expanded_height = height + 2 * expansion

    return expanded_x, expanded_y, expanded_width, expanded_height

def predict(image):
    # Preprocess the image
    resized_img = cv2.resize(image, (200, 200))
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    # print(gray_img.shape)
    gray_img = np.expand_dims(gray_img, axis=0)
    gray_img = np.expand_dims(gray_img, axis=-1)
    # Perform prediction
    prediction = model.predict(gray_img)
    age = np.round(prediction[0][0][0])

    if prediction[1] > 0.5:
        gender = 'Female'
    else:
        gender = 'Male'

    return age, gender

def extract_face(image):
    # Detect faces using MTCNN
    faces = mtcnn_model.detect_faces(image)
    all_faces = []
    for face in faces:
        x, y, width, height = face['box']
        x, y, width, height = expand_bounding_box(x, y, width, height, 10)
        crop_img = image[y:y + height, x:x + width]
        all_faces.append(crop_img)

    return all_faces

def call():
    container = st.container()
    container.title("Age and Gender Prediction")
    container.text("Upload an image and get the prediction")

    # Upload image
    uploaded_file = container.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)

        # Convert PIL image to NumPy array
        image = np.asarray(image)
        print(image.shape)
        height = image.shape[0]
        width = image.shape[1]
        if height > 480: height = int(image.shape[0] * 0.25)
        if width > 640: width = int(image.shape[1] * 0.25)
        # max_height =  int(image.shape[0] * 0.5)
        max_height =  height
        # max_width = int(image.shape[1] * 0.5)
        max_width = width
        reszied_img = cv2.resize(image, (max_width, max_height))
        container.image(reszied_img, caption='Uploaded Image', use_column_width=False)

        # Perform prediction
        if container.button('Predict'):
            faces = extract_face((image))
            i = 1
            for img in faces:
                age, gender = predict(img)
                img = cv2.resize(img,(100,100))
                container.image(img, caption='Extracted face: '+ str(i), use_column_width=False)
                i+=1
                # Display prediction results
                container.text(f"Predicted Age: {int(age)} yrs")
                container.text(f"Predicted Gender: {gender}")


if __name__ == '__main__':
    call()