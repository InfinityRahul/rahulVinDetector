import numpy as np
import streamlit as st
try:
 from PIL import Image
except ImportError:
 import Image
from vininfo import Vin
import cv2
import pytesseract


def import_and_predict(image_data):
    
        strinvar = pytesseract.image_to_string(image_data)
        
        return strinvar

st.title("""
          Vin/Number Plate Detection And Decoding
         """
         )
        
##pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\rahulbhatt\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
##pytesseract.pytesseract.tesseract_cmd = ‘/app/.apt/usr/bin/tesseract’

file = st.file_uploader("Please upload an image of the relevant category", type=["jpg", "png","JPEG"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    img_cv = cv2.imread(file.name)
    img_cvv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
##Noise reduction    
    gray = cv2.cvtColor(img_cvv, cv2.COLOR_RGB2GRAY)
    gray, img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = cv2.bitwise_not(img_bin)
    kernel = np.ones((2, 1), np.uint8)
    img = cv2.erode(gray, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    
##Calling priction procedure
    prediction = import_and_predict(img)
    

    if st.button('Predict'):
        st.write(prediction)
        
        
