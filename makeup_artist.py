from keras.preprocessing import image
import cv2
face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_alt.xml')
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5')
model._make_predict_function()
import numpy as np
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

class Makeup_artist(object):
    def __init__(self):
        pass

    def apply_makeup(self, img):
        faces = face_cascade.detectMultiScale(img, 1.3, 4) 
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 1)
            detected_face = img[int(y):int(y+h), int(x):int(x+w)]
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
            detected_face = cv2.resize(detected_face, (48, 48))
            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            emotion = emotions[max_index]
            cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return img