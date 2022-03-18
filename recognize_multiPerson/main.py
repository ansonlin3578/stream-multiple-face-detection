from pdb import line_prefix
import cv2
import imutils
import numpy as np
import tensorflow.keras
import  scipy
import scipy.spatial as T
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.layers import Input
from keras import Model
from tkinter import * 
from PIL import ImageTk, Image
#-------------------------------------------------------------
#tkinter
root = Tk()
# Create a frame
app = Frame(root, bg="white")
app.grid()
# Create a label in the frame
lmain = Label(app)
lmain.grid()

# 開啟影片檔案
cap = cv2.VideoCapture(0)

# vggface 的人臉特徵取得
vgg_model = VGGFace()
out = vgg_model.get_layer('fc7').output
model = Model(vgg_model.input, out)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                'haarcascade_frontalface_default.xml')

feats_path = 'feats.txt'
label_path = 'feats_label.txt'
face_number = 41
anson_feats = np.zeros((face_number,4096))
feat_index = np.zeros((face_number))

# 讀取dataset_feats檔案
def readFile(feats_path):
  fileObj = open(feats_path, "r") #opens the file in read mode
  words = fileObj.read().splitlines() #puts the file into an array
  k = 0
  for i in range(4096 * face_number):
    if(i >= 4096):
      k = (i % 4096)    #index from "0"
      anson_feats[(i // 4096)][k] = words[i]
    if(i < 4096):
      anson_feats[0][i] = words[i]
  fileObj.close()
  return anson_feats
readFile(feats_path)

feats_list = np.zeros((face_number,1,4096))
for i in range(face_number):
  for j in range(4096):
    feats_list[i][0][j] += anson_feats[i][j]#轉換成三維陣列
print(feats_list)
print(feats_list.shape)
print(feats_list[0])
print(feats_list[0].shape)

# 讀取dataset_feat Label檔案
def ReadIndex(label_path):
  fileObj = open(label_path, "r") #opens the file in read mode
  words = fileObj.read().splitlines() #puts the file into an array
  for i in range(face_number):
    feat_index[i] = words[i]
    #print(int(feat_index[i]))
ReadIndex(label_path)
# for i in range(face_number):
  

print("--------------------")

#萃取串流影像的特徵向量
def extract(model, frame):
    x = np.expand_dims(frame, axis=0)
    x = x.astype('float32')
    x = utils.preprocess_input(x)
    feats = model.predict(x)
    return feats

# 以迴圈從影片檔案讀取影格，並顯示出來
while(cap.isOpened()):
  ret, frame = cap.read()
  if(ret):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if(faces == ()):
      print("No Faces")

    face = []
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      face = np.array(frame[x:x+w, y:y+h])
      print("-------------------------")

      text = ""
      idx = None
      def showText(idx):
        if int(idx) == 0:
          return "Anson Lin"
        elif int(idx) == 1:
          return "Bowey Chen"
        elif int(idx) == 2:
          return "Ingwen Tsai"
        else:
          return None 

      if(face.shape[0] != 0):
        #print(face.shape)
        face = cv2.resize(face, (224, 224), interpolation = cv2.INTER_AREA)
        feats_stream = extract(model, face)
    
        for i in range(face_number):
          result = T.distance.cosine(feats_stream, feats_list[i])
          #print(result)
          idx = feat_index[i]
          if(0 <= result <= 0.1):
            print(idx)
            text = showText(idx)
            break
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                  0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # 顯示結果
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    
cap.release()
cv2.destroyAllWindows()
#------------------------------------------------------------------------


