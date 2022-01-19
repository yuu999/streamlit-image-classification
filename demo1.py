import streamlit as st
import cv2 
import numpy as np


uploaded_file = st.file_uploader("Upload image file first...", type=['jpg'])


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    x = st.sidebar.slider('Change Threshold value',min_value = 50,max_value = 255)
    ret,thresh1 = cv2.threshold(image_gray,x,255,cv2.THRESH_BINARY)
    thresh1 = thresh1.astype(np.float64)
    thresh2 = cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image_gray, caption='Gray Image', clamp=True)

    with col2:
        st.image(thresh1, caption='Thresh Binary', clamp=True)

    with col3:
        st.image(thresh2, caption='Thresh Adaptive', clamp=True)
