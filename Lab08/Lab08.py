import cv2
import numpy

cam=cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

dist = numpy.mat([-0.12753098, 0.597522, -0.00487466, -0.03123174, -0.92187974])
inti = numpy.mat([[728.4829254, 0, 260.52728928], [0,721.60506926,201.66307038],[0, 0, 1]])

objpoints=numpy.array([[0,0,0],[0.9,0,0],[0,1.9,0],[0.9,1.9,0]], dtype=numpy.float64)
objpoints2=numpy.array([[0,0,0],[0.19,0,0],[0,0.19,0],[0.19,0.19,0]], dtype=numpy.float64)

while True:

    ret, img = cam.read()
    rects, weights = hog.detectMultiScale(img,winStride=(4,4),padding=(16, 16), scale=1.04,useMeanshiftGrouping = False)
    faces = face_cascade.detectMultiScale(img,scaleFactor=1.2,minNeighbors=3)

    for (x, y, w, h) in rects:
        prin =0
        for (r,t,p,u) in faces:
            if x<r and y<t and x+w>r+p and y+h>t+u:
                prin = 1
        if prin==1:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            corners=numpy.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]], dtype=numpy.float64)
            ret, rvec, tvec = cv2.solvePnP(objpoints, corners, inti, dist)
            cv2.putText(img, "z: %.2f" % ((tvec[2])), (x,y ), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))

    for (x, y, w, h) in faces:
        corners=numpy.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]], dtype=numpy.float64)
        ret, rvec, tvec = cv2.solvePnP(objpoints2, corners, inti, dist)
        cv2.putText(img, "z: %.2f" % ((tvec[2])), (x,y ), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('camera',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
       
cam.release()
cv2.destroyAllWindows()
