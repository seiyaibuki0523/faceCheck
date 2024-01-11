import cv2
import numpy as np

faceXml = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recong = cv2.face.FaceRecognizer
faces = []
ids = []

a = int(input())
NUMBER_OF_PHOTOS = 3
for i in range(1,a+1):
    for j in range(1, NUMBER_OF_PHOTOS+1):
        img = cv2.imread(f'../face{i}/{j}.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
        img_np = np.array(gray, np.uint8)
        face = faceXml.detectMultiScale(gray)
        for detected in face:
            x,y,w,h = detected
            faces.append(img_np[y:y+h,x:x+w])
            ids.append(j)
        

print("traing...")
recong.train(faces,np.array(ids))
recong.save("face.yml")
print("OK")
