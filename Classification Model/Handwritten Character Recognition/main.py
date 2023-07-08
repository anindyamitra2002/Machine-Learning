import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
from keras.models import load_model
# Set the canvas size
CANVAS_SIZE = 250
ext = 10
threshold_value = 120
max_val = 255

classifier = load_model(r'OCR_e40_23_93_v4.h5')

st.set_page_config(
        page_title="Handwritten ChaRecognizer",
        page_icon="✒️")

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .canvas-container {
                    border: 7px solid #dddddd;
                    border-radius: 5px;
               }
        </style>
        """, unsafe_allow_html=True)

def remove_whitespace(img):
  try:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(img, threshold_value, max_val, cv2.THRESH_BINARY)
    gray = 255*(image < 128).astype(np.uint8) # To invert the text to white
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)) # Perform noise filtering
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    rect = image[y-ext:y+h+ext, x-ext:x+w+ext] # Crop the image - note we do this on the original image
    # print(rect.shape)
    rect = cv2.resize(rect, (224,224))
  except:
    # print(image.shape)
    rect = cv2.resize(image, (224,224))

  rect = np.expand_dims(rect, axis=-1)
  rect = np.expand_dims(rect, axis=0)
  return rect

def main():


    st.title("Handwritten Character Classification")
    # Create a canvas to draw on
    canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",  # Background color
        stroke_width=10,
        stroke_color="#000000",
        background_color="#ffffff",
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas",
    )
    btn = st.button("Predict")
    st.header("Predictions:")
    if btn:
        if canvas.image_data is not None:
            img = Image.fromarray(canvas.image_data.astype("uint8"))
            # st.image(img, caption="Saved Drawing", use_column_width=True)
            image = np.array(img)
            image = image[:,:,:3]
            image = remove_whitespace(image)
            y_pred = classifier.predict(image)
            idx = np.argmax(y_pred)
            acc = np.max(y_pred)
            if idx > 9:
                st.subheader(f"Label: {chr(idx - 10 + 65)}")
                st.subheader(f":green[Accuracy:] {acc*100.0}%")
            else:
                st.subheader(f"Label: {idx}")
                st.subheader(f":green[Accuracy:] {acc*100.0}%")





if __name__ == "__main__":
    main()
