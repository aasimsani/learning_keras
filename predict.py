
from tensorflow.keras import models
import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical

from helper import rgb2gray,data

import cv2
import dlib



# Load stored model
path = "./models/model-1542912426.7963462.h5"
new_model = models.load_model(path)


# Use dlib to get facebox 
detector = dlib.get_frontal_face_detector()
camera_port = 0
 
# Setup camera input
# Can also just use video file
camera = cv2.VideoCapture(camera_port)
 
def get_image():
    retval, im = camera.read()
    return im

while(True):

    # Capture frame-by-frame
    ret, frame = camera.read()

    # If no frames break
    if ret == False:
        break

    # Find face
    dets = detector(frame, 0)

    # For each face 
    for i, d in enumerate(dets):

        # Crop the image to the face
        crop = frame[d.top():d.bottom(), d.left():d.right()]

        # Resize for CNN
        newimg = cv2.resize(crop,(64,64))
        
        # Turn to grayscale
        gray = rgb2gray(np.array(newimg))

        # Reshape for CNN input
        reshape = np.reshape(gray,(1,64,64,1))

        # Predict output and print informative prompt
        prediction = new_model.predict(reshape)

        if prediction[0][1] > 0.85:
            print("Smiling")
        elif prediction[0][0] > 0.85:
            print("Not smiling")
        else:
            print("Unsure")
        
        # Show image
        cv2.imshow('frame',frame)


    # Press 'q' to stop video 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Taking image...")
        image = frame
        break

# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
del(camera)


