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

	inti = numpy.mat([[916.09153502, 0, 464.53194172],[0, 896.42432491, 375.07133942],[0, 0, 1]])
	dis = numpy.mat([0.05268609, -0.69896519, 0.02428493, -0.00895138, -0.42484144])

	drone.takeoff()

	while(1):
		frame = drone.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

		markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
	    #print rejectedCandidates
	    #print markerCorners
		cv2.imshow('gray',gray)

		if cv2.waitKey(1) & 0xFF == ord('q'):
		    break

		key = cv2.waitKey(1)

		hw=14.5
		if key!= -1:
			drone.keyboard(key)
		elif len(markerCorners)>0:
			try:
				if markerIds!=None:
					print("one entry")
					print(markerIds)
					if markerIds==1:
						hw=13.8
			except:
				print (markerIds)
				for i in range(len(markerIds)):
					if markerIds[i]==1:
						markerIds=numpy.ndarray([1],int)
						markerCorners=[(markerCorners[i])]
						print (markerIds)
						break
			frame = cv2.aruco.drawDetectedMarkers(gray, markerCorners, markerIds)
			rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, hw, inti, dis)
			frame2 = cv2.aruco.drawAxis(frame, inti, dis, rvec, tvec, 7)
			cv2.putText(frame2, "x %.1f y %.1f z %.1f -- %.0f deg" % ((tvec[0][0][0]), (tvec[0][0][1]), (tvec[0][0][2]), (rvec[0][0][2] / math.pi * 180)), (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (244, 244, 244))
		    # print _objPoints
			cv2.imshow('g',frame2)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		    # cv2.putText()
			z = [[0],[0],[1]]
			r = cv2.Rodrigues(rvec)
			z1 = -numpy.dot(numpy.array(r[0]),numpy.array(z))
			z1[1] = 0
			v = z1
			radians = math.atan2(v[0], v[2])
			angle = math.degrees(radians)
			# print (angle)
			# print ("")
			try:
				if markerIds!=None and markerIds==1:
					if numpy.abs(angle)>30:
						if angle<0:
							drone.rotate_ccw(numpy.abs(30))
						else:
							drone.rotate_cw(numpy.abs(30))
					else:
						print(tvec[0][0][1])
						print(tvec[0][0][0])
						if tvec[0][0][1]>5:
							drone.move_down(0.2)
						elif tvec[0][0][1]<-30:
							drone.move_up(0.2)
						elif tvec[0][0][0]>20:
							drone.move_right(0.2)
						elif tvec[0][0][0]<-20:
							drone.move_left(0.2)
						else:
							if (tvec[0][0][2])>110:
								drone.move_forward(0.4)
							elif (tvec[0][0][2])<80:
								drone.move_backward(0.2)
							else:
								drone.move_forward(0.2)
				elif markerIds!=None and markerIds!=1:
					print (markerIds, tvec[0][0][2])
					if (tvec[0][0][2])<= 55:
						drone.move_backward(0.2)
					elif abs(tvec[0][0][2]-60)<5 and markerIds == 4:
						drone.land()
						break
					elif abs(tvec[0][0][2]-60)<5 and markerIds == 11:
						drone.rotate_cw(90)
					else:
						if (tvec[0][0][2])>65:
							if numpy.abs(angle)>15 and tvec[0][0][2] < 150:
								if angle<0:
									drone.rotate_ccw(numpy.abs(10))
								else:
									drone.rotate_cw(numpy.abs(10))
							elif numpy.abs(angle)>30 and tvec[0][0][2] >= 150:
								if angle<0:
									drone.rotate_ccw(numpy.abs(20))
								else:
									drone.rotate_cw(numpy.abs(20))
							elif tvec[0][0][1]>5:
								drone.move_down(0.2)
							elif tvec[0][0][1]<-60 and tvec[0][0][2]>150:
								drone.move_up(0.2)
							elif tvec[0][0][1]<-30 and tvec[0][0][2]<=150:
								drone.move_up(0.2)
							elif tvec[0][0][0]>10:
								drone.move_right(0.2)
							elif tvec[0][0][0]<-10:
								drone.move_left(0.2)
							else:
								if (tvec[0][0][2])>120:
									drone.move_forward(0.5)
								elif markerIds == 4 and tvec[0][0][2]>80:
									drone.move_forward(min(tvec[0][0][2]/100-0.6, 0.25))
								else:
									drone.move_forward(0.2)
						# else:
						# 	drone.move_backward(0.2)
			except:
				print ("error")
			cv2.imshow('gray',frame2)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break


if __name__ == "__main__":
    main()
