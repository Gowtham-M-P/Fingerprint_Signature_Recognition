import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import sys

#load the Trained model
model = load_model('signature_weights.h5')

def detect(img):
    IMAGE_RESIZE = 224
    CATEGORIES = ["Anil","Ashoka","Karthik","Manoj","Pawan","Prajwal","Sagar","Shubham","Souparnika","Srihari","Vignesh","Vinay"]
    image = cv2.imread(img)
    resized = cv2.resize(image, (IMAGE_RESIZE,IMAGE_RESIZE), interpolation=cv2.INTER_AREA)
    resized = resized.reshape(1,IMAGE_RESIZE,IMAGE_RESIZE,3)
    predicted = model.predict(resized)
    print(CATEGORIES[np.argmax(predicted)])

    return

if __name__ == "__main__":
    imgPath = sys.argv[1]
    detect("/images/Ashoka/signature_0_285.jpeg")