import os
import cv2

src = "./Pokemon" 
dst = "./resizedData" 

for each in os.listdir(src):
    img = cv2.imread(os.path.join(src,each))
    img = cv2.resize(img,(32,32))
    cv2.imwrite(os.path.join(dst,each), img)
