import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    # Use Image.Resampling.LANCZOS instead of Image.ANTIALIAS
    image = ImageOps.fit(image_data, (100, 100), Image.Resampling.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)

    # Display the image with a custom width
    st.image(image, channels='RGB', width=500)  # Set width to 500 pixels for larger display

    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Load your pre-trained model
model = tf.keras.models.load_model('my_model2.h5')

# Streamlit app title and description
st.image('logo.png', caption='Glaucoma Prediction', use_column_width=True)
st.write("""
         # ***Glaucoma detector***
         """)

st.write("This is a simple image classification web app to predict glaucoma through fundus image of the eye.")

# Image file uploader
file = st.file_uploader("Please upload an image (jpg) file", type=["jpg"])

if file is None:
    st.text("You haven't uploaded a jpg image file.")
else:
    imageI = Image.open(file)
    prediction = import_and_predict(imageI, model)
    pred = prediction[0][0]
    
    # Display prediction
    if pred > 0.5:
        st.write("""
                 ## **Prediction:** Your eye is Healthy. Great!!
                 """)
        st.balloons()
    else:
        st.write("""
                 ## **Prediction:** You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.
                 """)
