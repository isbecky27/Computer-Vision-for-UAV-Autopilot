import tello
import cv2
import numpy
from tello_control_ui import TelloUI
import time
import math

def main():
	drone = tello.Tello('', 8889)

	time.sleep(5)

	dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
	parameters =  cv2.aruco.DetectorParameters_create()

	# data.txt
	inti = numpy.mat([[916.09153502, 0, 464.53194172],[0, 896.42432491, 375.07133942],[0, 0, 1]])
	dis = numpy.mat([0.05268609, -0.69896519, 0.02428493, -0.00895138, -0.42484144])

	drone.takeoff()
	
	while(1):
		frame = drone.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


		markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
	    #print rejectedCandidates
	    #print markerCorners
		cv2.imshow('g',gray)

		if cv2.waitKey(1) & 0xFF == ord('q'):
		    break

		if len(markerCorners)>0:
			frame = cv2.aruco.drawDetectedMarkers(gray, markerCorners, markerIds)
			rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 13.8, inti, dis)
			frame2 = cv2.aruco.drawAxis(frame, inti, dis, rvec, tvec, 1)
			cv2.putText(frame2, "%.1f cm -- %.0f deg" % ((tvec[0][0][2]), (rvec[0][0][2] / math.pi * 180)), (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (244, 244, 244))
		    #print _objPoints

		    #cv2.putText()
			z = [[0],[0],[1]]
			r = cv2.Rodrigues(rvec)
			z1 = -numpy.dot(numpy.array(r[0]),numpy.array(z))
			z1[1] = 0
			v = z1
			radians = math.atan2(v[0], v[2])
			angle = math.degrees(radians)
			# print (angle)
			# print ("")
			if numpy.abs(angle/2)>20:
				if angle<0:
					drone.rotate_ccw(numpy.abs(angle/3))
				else:
					drone.rotate_cw(numpy.abs(angle/3))
			else:	
				if (tvec[0][0][2])<70:
					drone.move_backward(0.2)
				else:
					drone.move_forward(0.2)
			cv2.imshow('g',frame2)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break


		key = cv2.waitKey(1)


		if key!= -1:
			drone.keyboard(key)


if __name__ == "__main__":
    main()
