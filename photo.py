from keras.models import model_from_json
from keras_preprocessing.image import load_img
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

json_file = open("model/autistic_emotion_recognition_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model/autistic_emotion_recognition_model.h5")

label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def ef(image):
    img = load_img(image, color_mode='grayscale', target_size=(48,48))
    feature = np.array(img)
    feature = feature.reshape(48, 48)
    return feature / 255.0



image_path = ''

img = ef(image_path)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)


