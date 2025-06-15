import cv2
import glob


images = [cv2.imread(file) for file in glob.glob("sample\\*.jpg")]
print(len(images))
pictPath = r'classifier\cascade.xml'
face_cascade = cv2.CascadeClassifier(pictPath) 

for i in range(146):
   img = cv2.resize(images[i], (0, 0), fx = 0.3, fy = 0.21)
   faces = face_cascade.detectMultiScale(img, scaleFactor=1.07,
      minNeighbors=6, minSize=(25,25),maxSize=(200,200)) 

   for (x,y,w,h) in faces:
      plate = img[y: y+h, x:x+w]
      plate = cv2.blur(plate,ksize=(70,20))
      img[y: y+h, x:x+w] = plate
   savePath = "result\\result"+str(i)+".jpg"
   cv2.imwrite(savePath, img)