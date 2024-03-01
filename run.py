import cv2
import tensorflow as tf
import numpy as np

# Инициализация
# модель
model_path = "model/autistic_emotion_recognition_model.json"
# веса
weights_path = "model/autistic_emotion_recognition_model.h5"
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# количество выборок
n_samples = 10


model = tf.keras.models.load_model(model_path)

labels = {n:cl for n,cl in enumerate(['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])}

def vectorize_image(image):
    feature = np.array(image)
    feature = feature.reshape(1,64,64,3)
    return feature

queue = np.zeros((n_samples, 7))    

face_cascade = cv2.CascadeClassifier(haar_file)
webcam=cv2.VideoCapture(0)
while True:
    ret, frame = webcam.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    try: 
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pred = model.predict(vectorize_image(face))
            
            queue[:-1] = queue[1:]
            queue[-1] = pred
            
            prediction_label = labels[np.sum(queue, axis=0).argmax()]
            cv2.putText(frame, '% s' %(prediction_label), (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        cv2.imshow("Output", frame)
        k = cv2.waitKey(30) & 0xFF
        if k==27:
            break
    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()