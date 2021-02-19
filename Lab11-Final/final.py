import tello
import cv2
import numpy as np
from tello_control_ui import TelloUI
import time
import math
import os



def main():
	fl=0
	left=0
	drone = tello.Tello('', 8889)

	time.sleep(5)

	dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
	parameters =  cv2.aruco.DetectorParameters_create()


	inti = np.mat([[916.09153502, 0, 464.53194172],[0, 896.42432491, 375.07133942],[0, 0, 1]])
	dist = np.mat([0.05268609, -0.69896519, 0.02428493, -0.00895138, -0.42484144])
	objpoints2=np.array([[0,0,0],[0.20,0,0],[0,0.21,0],[0.20,0.21,0]], dtype=np.float64)
	objpointse=np.array([[0,0,0],[0.19,0,0],[0,0.15,0],[0.19,0.15,0]], dtype=np.float64)
	objpointsb=np.array([[0,0,0],[0.14,0,0],[0,0.16,0],[0.14,0.16,0]], dtype=np.float64)
	objpointst=np.array([[0,0,0],[0.1,0,0],[0,0.21,0],[0.10,0.21,0]], dtype=np.float64)

	args={}
	args['threshold'] = 0.3
	args['yolo'] = 'yolo-coco'
	args['confidence'] = 0.5

	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
		dtype="uint8")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
	configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	drone.takeoff()
	action = 0
	while(1):
		print('left', left)
		key=cv2.waitKey(1)
		if key!= -1:
			drone.keyboard(key)
			continue
		image = drone.read()
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		#
		# ret,image=cam.read()
		# load our input image and grab its spatial dimensions
		# image = cv2.imread(args["image"])
		(H, W) = image.shape[:2]

		# determine only the *output* layer names that we need from YOLO

		if fl==0:
			ln = net.getLayerNames()
			ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

			# construct a blob from the input image and then perform a forward
			# pass of the YOLO object detector, giving us our bounding boxes and
			# associated probabilities
			blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			start = time.time()
			layerOutputs = net.forward(ln)
			end = time.time()

			# show timing information on YOLO
			print("[INFO] YOLO took {:.6f} seconds".format(end - start))

			# initialize our lists of detected bounding boxes, confidences, and
			# class IDs, respectively
			boxes = []
			confidences = []
			classIDs = []

			# loop over each of the layer outputs
			for output in layerOutputs:
				# loop over each of the detections
				for detection in output:
					# extract the class ID and confidence (i.e., probability) of
					# the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
					if confidence > args["confidence"]:
						# scale the bounding box coordinates back relative to the
						# size of the image, keeping in mind that YOLO actually
						# returns the center (x, y)-coordinates of the bounding
						# box followed by the boxes' width and height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top and
						# and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates, confidences,
						# and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)

			# apply non-maxima suppression to suppress weak, overlapping bounding
			# boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
				args["threshold"])

			# ensure at least one detection exists
			p=0
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])

					# draw a bounding box rectangle and label on the image
					color = [int(c) for c in COLORS[classIDs[i]]]
					if LABELS[classIDs[i]]=='horse':
						p=1
						corners=np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]], dtype=np.float64)
						ret, rvec, tvec = cv2.solvePnP(objpoints2, corners, inti, dist)
						cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
						text = "{}: {:.4f}\{}".format(LABELS[classIDs[i]], confidences[i],(tvec[2]))
						cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
						tvec[2]=tvec[2]*100
						tvec[1]=tvec[1]*100
						tvec[0]=tvec[0]*100
						if (tvec[2])<30:
							drone.move_backward(0.2)
						elif tvec[2]<60:
							print("it's me!!")
							time.sleep(3)
							drone.move_right(0.7)
							fl=1
							p=0

						else:
							print(tvec[0])
							if (tvec[2])>60:
								# if tvec[1]>5:
								# 	drone.move_down(0.2)
								# elif tvec[1]<-60 and tvec[2]>150:
								# 	drone.move_up(0.2)
								# elif tvec[1]<-30 and tvec[2]<=150:
								# 	drone.move_up(0.2)
								# else:
								if (tvec[2])>120:
									drone.move_forward(1)
								elif tvec[0]>25:
									drone.move_right(0.2)
								elif tvec[0]<-25:
									drone.move_left(0.2)
								elif tvec[2]>80:
									drone.move_forward(min(tvec[2]/100-0.6, 0.25))
								else:
									drone.move_forward(0.2)
					elif LABELS[classIDs[i]]=='banana' or LABELS[classIDs[i]]=='elephant':
						if LABELS[classIDs[i]]=='banana':
							corners=np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]], dtype=np.float64)
							ret, rvec, tvec = cv2.solvePnP(objpointsb, corners, inti, dist)
							cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
							text = "{}: {:.4f}\{}".format(LABELS[classIDs[i]], confidences[i],(tvec[2]))
							cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
							tvec[2]=tvec[2]*100
							tvec[1]=tvec[1]*100
							tvec[0]=tvec[0]*100
						else:
							corners=np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]], dtype=np.float64)
							ret, rvec, tvec = cv2.solvePnP(objpointse, corners, inti, dist)
							cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
							text = "{}: {:.4f}\{}".format(LABELS[classIDs[i]], confidences[i],(tvec[2]))
							cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
							tvec[2]=tvec[2]*100
							tvec[1]=tvec[1]*100
							tvec[0]=tvec[0]*100

						if (tvec[2])>80:
							p=1
							drone.move_forward(0.2)
						elif (tvec[2])<50:
							p=1
							drone.move_backward(0.2)
					elif LABELS[classIDs[i]]=='traffic light':
						corners=np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]], dtype=np.float64)
						ret, rvec, tvec = cv2.solvePnP(objpointst, corners, inti, dist)
						cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
						text = "{}: {:.4f}\{}".format(LABELS[classIDs[i]], confidences[i],(tvec[2]))
						cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
						tvec[2]=tvec[2]*100
						tvec[1]=tvec[1]*100
						tvec[0]=tvec[0]*100
						drone.flip('b')
						time.sleep(1)
						if tvec[0]>20:
							drone.move_right(0.2)
						elif tvec[0]<-20:
							drone.move_left(0.2)
						time.sleep(1)
						if (tvec[2])>80:
							drone.move_forward(0.2)
						elif (tvec[2])<50:
							drone.move_backward(0.2)
						time.sleep(3)
						drone.move_left(0.8)
						fl=1



		if p==1:
			cv2.imshow('g',image)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			continue
		cv2.imshow('g',image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


		markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

		key = cv2.waitKey(1)


		hw=14.5
		print('p',p)

		if key!= -1:
			drone.keyboard(key)
		elif len(markerCorners)>=0:
			try:
				if markerIds!=None:
					print("one entry")
					print(markerIds)
				else:
					print("None")
			except:
				print(markerIds)
				for i in range(len(markerIds)):
					print markerIds[i]
					if markerIds[i]==4:
						markerIds=np.array([4],int)
						markerCorners=[(markerCorners[i])]
						print (markerIds,'asds')
						break
					elif markerIds[i]==14:
						markerIds=np.array([14],int)
						markerCorners=[(markerCorners[i])]
						print (markerIds,'asds')
						break
			rvec=None
			tvec=None
			try:
				frame = cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)
				rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, hw, inti, dist)
				frame2 = cv2.aruco.drawAxis(frame, inti, dist, rvec, tvec, 7)
				cv2.putText(frame2, "x %.1f y %.1f z %.1f -- %.0f deg" % ((tvec[0][0][0]), (tvec[0][0][1]), (tvec[0][0][2]), (rvec[0][0][2] / math.pi * 180)), (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (244, 244, 244))
			    #print _objPoints
			except Exception as e:
				print (e)
				frame2=image
			cv2.imshow('g',frame2)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			try:
			    #cv2.putText()
				z = [[0],[0],[1]]
				r = cv2.Rodrigues(rvec)
				z1 = -np.dot(np.array(r[0]),np.array(z))
				z1[1] = 0
				v = z1
				radians = math.atan2(v[0], v[2])
				angle = math.degrees(radians)
				print 'here'
				if markerIds!=None and markerIds==4 :
					if (tvec[0][0][2])<= 35:
						drone.move_backward(0.2)
					elif tvec[0][0][2]<=55 and markerIds == 4:
						drone.move_left(0.85)
						time.sleep(3)
					else:
						if (tvec[0][0][2])>55:
							if tvec[0][0][1]>0: #
								drone.move_down(0.2)
							elif tvec[0][0][1]<-70 and tvec[0][0][2]>150:
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
									drone.move_forward(min(tvec[0][0][2]/100-0.5, 0.25))
								else:
									drone.move_forward(0.2)

				elif markerIds!=None and markerIds==11 :
					if (tvec[0][0][2])<= 110:
						drone.move_backward(0.2)
					elif tvec[0][0][2]<=140 and markerIds == 11:
						left=1
						time.sleep(3)
						drone.rotate_ccw(np.abs(180))
						time.sleep(3)
						fl=0

					else:
						drone.move_forward(0.7)
				elif markerIds!=None and markerIds!=21:
					left=0
					print (markerIds,tvec[0][0][0],tvec[0][0][2])
					if (tvec[0][0][2])<= 55:
						drone.move_backward(0.2)
					elif abs(tvec[0][0][2]-60)<5 and markerIds == 22:
						drone.land()
						break
					elif abs(tvec[0][0][2]-60)<8 and markerIds == 14:
						drone.rotate_ccw(90)
						time.sleep(3)
					elif abs(tvec[0][0][2]-60)<5 and markerIds == 45:
						drone.rotate_cw(90)
						time.sleep(3)
					else:
						if (tvec[0][0][2])>65:
							if np.abs(angle)>15 and tvec[0][0][2] < 150:
								if angle<0:
									drone.rotate_ccw(np.abs(10))
								else:
									drone.rotate_cw(np.abs(10))
							elif np.abs(angle)>30 and tvec[0][0][2] >= 150:
								if angle<0:
									drone.rotate_ccw(np.abs(20))
								else:
									drone.rotate_cw(np.abs(20))
							elif tvec[0][0][1]>0: #
								drone.move_down(0.2)
							elif tvec[0][0][1]<-120 and tvec[0][0][2]>200:
								drone.move_up(0.2)
							elif tvec[0][0][1]<-60 and tvec[0][0][2]>150:
								drone.move_up(0.2)
							elif tvec[0][0][1]<-30 and tvec[0][0][2]<=150:
								drone.move_up(0.2)
							elif tvec[0][0][0]>10:
								drone.move_right(0.2)
							elif tvec[0][0][0]<-10:
								drone.move_left(0.2)
							else:
								if (tvec[0][0][2])>150:
									drone.move_forward(0.8)
								elif tvec[0][0][2]>80:
									drone.move_forward(min(tvec[0][0][2]/100-0.6, 0.25))
								else:
									drone.move_forward(0.2)

			except Exception as e:
				print (e)
			if left==1:
				print('left')
				drone.move_left(0.3)
			cv2.imshow('g',frame2)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break



if __name__ == "__main__":
    main()
