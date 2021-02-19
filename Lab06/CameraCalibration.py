import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0) # device
objp = np.zeros( (7*5, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
objpoints = []
imgpoints = []


imgs = []
for idx in range(1, 15):
	file_name = 'drone/'+str(idx)+'.jpg'
	img = cv2.imread(file_name, 1)
	imgs.append(img)

# while True :
for frame in imgs:
	# ret, frame = cap.read()
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#cv2.imshow('frame', gray)

	ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)
	
	# find corner points
	if ret == True:
		corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
		(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
		
		objpoints.append(objp)
		imgpoints.append(corners2)
		#print("imgpoints is: ")
		#print(corners)
		#print("objpoints is: ")
		#print(objp)
		cv2.drawChessboardCorners(frame, (7, 5), corners, ret)
		
		cv2.imshow('img', frame)

		if cv2.waitKey(5000) & 0xFF == ord('q'):
			break

print("total images: {}".format( len(imgpoints)))
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("mtx is: ")	
print(mtx)
print()
print("dist is: ")
print(dist)
print()
	

f = cv2.FileStorage("data.txt", cv2.FILE_STORAGE_WRITE)
f.write("intrinsic", mtx)
f.write("distortion", dist)
f.release()
cap.release()

cv2.destroyAllWindows()