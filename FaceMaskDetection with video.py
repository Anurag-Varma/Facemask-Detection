import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import array as arr
from keras.preprocessing.image import img_to_array

#load model
model = model_from_json(open("fer_600000.json", "r").read())      #change the path accoring to files
#load weights
model.load_weights('fer_600000.h5')        #change the path accoring to files
video="C:/Users/panur/Downloads/1.mp4"     #change the path accoring to files

ret=1
flag=True
cap = cv2.VideoCapture(video)
while(ret!=0 and flag):

 ret, fm=cap.read()
 
 file = cv2.resize(fm, (128, 128))
 file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)

 roi = img_to_array(file)
 roi = np.expand_dims(roi, axis=0)
    
 preds=model.predict_classes(roi)[0]
 if preds==0:
     print("Mask")
 else:
     print("No Mask")
  
 cv2.imshow("Live Video", fm)

 k=cv2.waitKey(25)
 if k == 27:     #Press ESC to exit/stop
    ret=0        
    break

print("closed")
cap.release()   
cv2.destroyAllWindows()