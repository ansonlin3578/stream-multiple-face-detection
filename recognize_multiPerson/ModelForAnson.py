
from audioop import avg
from operator import index
import cv2
import imutils
import numpy as np
import tensorflow.keras
import  scipy
import scipy.spatial as T
import os
from tensorflow.keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.layers import Input
from keras import Model

DATADIR = "dataset"
person_feats = []
IMG_SIZE = 224
CATEGORIES = ["anson", "bowey", "ingwen"]
picture_numbers = 41

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                'haarcascade_frontalface_default.xml')

def create_person_feats():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)  #0 for anson , 1 for bowey , 3 for ingwen
        for person_img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,person_img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                person_feats.append([new_array, class_num])
            except Exception as e:
                pass

create_person_feats()

feats_index = []
for features, label in person_feats:
    feats_index.append(label)
feats_index = np.array(feats_index)
feats_index = np.expand_dims(feats_index, axis = 1)
print(feats_index.shape)
print(feats_index[0].shape)
for i in range(41):
    print(feats_index[i])

def extract(model, person_face):
    new_array = cv2.resize(person_face, (IMG_SIZE,IMG_SIZE))
    x = np.expand_dims(new_array, axis=0)
    x = x.astype('float32')
    #print("------------------------------------------")
    #print(x)            #shape = (1, 224, 224 , 3)
    x = utils.preprocess_input(x)
    feats = model.predict(x)
    return feats

vgg_model = VGGFace()
out = vgg_model.get_layer('fc7').output
model = Model(vgg_model.input, out)

person_faces = []
feats = []
for i in range(41):
    img = person_feats[i][0]
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    print(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        person_face = np.array(img[x:x+w, y:y+h])
        person_faces.append(person_face)
person_faces = np.array(person_faces, dtype=object)
print(person_faces.shape)   #total numbers of my face are detected
face_number = person_faces.shape[0]

for person_face in person_faces[0:]:
    feat = extract(model, person_face)
    feats.append(feat)

feats_total = np.array(feats)
print(feats_total.shape)
feats_list = np.zeros((face_number,4096))

for i in range(face_number):
    for j in range(4096):
        feats_list[i][j] += feats_total[i][0][j]
print(feats_list)
print(feats_list[0].shape)

feats_file = open("feats.txt","w")    
for row in feats_list[0:]:
    np.savetxt(feats_file, row)
feats_file.close()
print("feats saved...")

feats_label_file = open("feats_label.txt","w")   
for row in feats_index[0:]:
    np.savetxt(feats_label_file, row)
feats_label_file.close()
print("feats label saved...")


