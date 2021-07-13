
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

#load model
model = model_from_json(open("fer.json", "r").read())            #change the path accoring to files
#load weights
model.load_weights('fer.h5')                    #change the path accoring to files


detection_model_path="C:/Users/panur/.spyder-py3/FaceMaskDetection/cascadeH5.xml"    #change the path accoring to files
face_detection = cv2.CascadeClassifier(detection_model_path)                         

ret=1
flag=True
cap = cv2.VideoCapture(0)   #default 0 for webcam 
frameRate = cap.get(30)

while(cap.isOpened()):

 ret, fm=cap.read()

 fm = cv2.resize(fm, (224, 224))
 file = cv2.cvtColor(fm, cv2.COLOR_BGR2RGB)
 
 orig_frame = file
 frame = file
 faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
 
 if len(faces) :
    faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48),3)
    roi = frame.astype("float") / 255.0
    
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    
    preds=model.predict_classes(roi)[0]
    if preds==0:
      print("Mask worn")
      test='Mask worn'
    elif preds==1:
      print("Danger: No Mask")
      test='Danger: No Mask'
 
    cv2.putText(fm,test, (fX-15, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(fm, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
 
   
 cv2.imshow("Live Video", fm)

 k=cv2.waitKey(25)   #Press ESC to stop/exit
 if k == 27: 
    ret=0        
    break

print("closed")
cap.release()   
cv2.destroyAllWindows()