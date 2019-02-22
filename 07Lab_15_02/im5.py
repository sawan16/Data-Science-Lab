import cv2
import sys

imagePath = sys.argv[1]

cascpath = "C:\opencv\opencv\sources\data\haarcascades\haarcascade_profileface.xml"
print(cascpath)
face_cascade = cv2.CascadeClassifier(cascpath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 2,minSize = (30,30))

print("Found {0} faces!".format(len(faces)))
 
# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found" ,image)
cv2.waitKey(0)

