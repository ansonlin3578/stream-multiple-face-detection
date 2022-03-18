from pdb import line_prefix
import cv2
import imutils
import numpy as np
import tensorflow.keras
import scipy
import scipy.spatial as T
from tensorflow.keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.layers import Input
from keras import Model
from tkinter import * 
from PIL import ImageTk, Image
#-------------------------------------------------------------
# Dlib 的人臉偵測器
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

def video_stream():  
# 以迴圈從影片檔案讀取影格，並顯示出來
  if(cap.isOpened()):  
    ret, frame = cap.read() #frame size (480,640)
    # if(ret):
    #   cv2.waitKey(1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #print(faces)
    face = []
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      face = np.array(frame[x:x+w, y:y+h])
      #print("-------------------------")

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
          #print(i)
          idx = feat_index[i]
          if(0 <= result <= 0.1):
            #print(idx)
            text = showText(idx)
            break
      cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                  0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    #print(frame.shape)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream) 

# 開啟影片檔案
cap = cv2.VideoCapture(0)

#tkinter_Setting
root = Tk()
root.title("Recognize multiPerson")

# Create a label in the frame
lmain = Label(root)
lmain.pack(side = 'left', padx=10, pady=10)
root.config(cursor="arrow")
lmain.grid(row=0, column=0)

GUI_title = Label(root, text="Total people numbers in datasets : 3",
                  font=(14), relief="flat", justify='left')
GUI_title.grid(row=0, column=1, ipady=10, sticky=N)

PersonName = Label(root, text="1.Anson Lin\n2.Bowey Chen\n3.Ingwen Tsai",
                  font=(14), relief="sunken", justify='left', bg='white')
#PersonName.place(x = 695, y = 40, anchor=N)
PersonName.grid(row=0, column=1, pady=40, sticky=NW)

video_stream()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
#------------------------------------------------------------------------


