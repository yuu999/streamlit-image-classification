from logging import getLogger, info
import copy
import logging

import streamlit as st
from PIL import Image
import cv2 
import numpy as np
from streamlit import logger

from classification import predict


logger = getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    st.title('Streamlit + OpenCVサンプル')
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Image Processing', 'Face Detection', 'Object Detection')
    )
    
    # st.set_option('deprecation.showfileUploaderEncoding', False)
    uploaded_file = st.file_uploader("ファイルアップロード", type='jpg')
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
    
    if uploaded_file:
        if selected_box == 'Welcome':
            welcome(opencv_image) 
        if selected_box == 'Image Processing':
            image_processing(opencv_image)
        if selected_box == 'Face Detection':
            face_detection(opencv_image)
        if selected_box == 'Object Detection':
            object_detection(opencv_image)
    
    else:
        st.warning('### Welcome. Please upload any image (.jpg) first.')


def cv2pil(opencv_image):
    new_image = opencv_image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def welcome(opencv_image):
    
    st.header('Welcom!!')
    st.subheader('A simple app that shows different image processings.')
    st.subheader('You can choose the options from the select box in the sidebar.')
    
    st.image(opencv_image, channels="BGR")


def image_processing(opencv_image):
    st.header("Image Processing.")
    st.subheader("Thresholding, Edge Detection and Contours")
    
    image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # if st.button('See Sample Image'):
    #     image = Image.open('img/sample.jpg')
    #     st.image(image, use_column_width=True)

    x = st.slider('Change Threshold value',min_value = 50,max_value = 255)
    ret,thresh1 = cv2.threshold(image,x,255,cv2.THRESH_BINARY)
    thresh1 = thresh1.astype(np.float64)
    st.image(thresh1, use_column_width=True,clamp = True)
    
    st.text("Bar Chart of the image")
    histr = cv2.calcHist([image],[0],None,[256],[0,256])
    st.bar_chart(histr)
    
    st.text("Press the button below to view Canny Edge Detection Technique")
    if st.button('Canny Edge Detector'):
        edges = cv2.Canny(opencv_image,50,300)
        st.image(edges,use_column_width=True,clamp=True)

    y = st.slider('Change Value to increase or decrease contours',min_value = 50,max_value = 255)     
    
    if st.button('Contours'):
          
        imgray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,y,255,0)
        image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        img = cv2.drawContours(opencv_image, contours, -1, (0,255,0), 3)
 
        st.image(thresh, use_column_width=True, clamp = True)
        st.image(img, use_column_width=True, clamp = True)


def face_detection(opencv_image):
    st.header("Face Detection using haarcascade")
  
    img_rgb_ = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    img_gray_ = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    
    faces = face_cascade.detectMultiScale(img_gray_)
    logger.info(f"{len(faces)} faces detected in the image.")
    for x, y, width, height in faces:
        cv2.rectangle(img_rgb_, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=2)
   
    st.image(img_rgb_, use_column_width=True,clamp = True)

    if st.button('Eyes Detection'):
        st.subheader("Detecting eyes from an image")

        eye = cv2.CascadeClassifier('data/haarcascade_eye.xml')  
        found = eye.detectMultiScale(img_gray_, minSize =(20, 20)) 
        amount_found_ = len(found)
            
        if amount_found_ != 0:
            img_rgb_copy_ = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            for (x, y, width, height) in found:
            
                cv2.rectangle(img_rgb_copy_, (x, y),  
                                (x + height, y + width),  
                                (0, 255, 0), 5) 
            st.image(img_rgb_copy_, use_column_width=True,clamp = True)

        else:
            st.subheader('Eyes were not detected!!')


# def object_detection(opencv_image):
    
#     st.header('Object Detection')
#     st.subheader("Object Detection is done using different haarcascade files.")
#     st.image(opencv_image, channels="BGR")
#     img_gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY) 
#     img_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB) 

#     clock = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')  
#     found = clock.detectMultiScale(img_gray, minSize=(20, 20)) 
#     amount_found = len(found)
#     st.text("Detecting a clock from an image")
#     if amount_found != 0:  
#         for (x, y, width, height) in found:
     
#             cv2.rectangle(img_rgb, (x, y),  
#                           (x + height, y + width),  
#                           (0, 255, 0), 5) 
#     st.sub(img_rgb, use_column_width=True, clamp = True)


def object_detection(opencv_image):

    st.header("Image Classification By ResNet50")
    img = cv2pil(opencv_image)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    labels = predict(img)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])


if __name__ == "__main__":
    main()
