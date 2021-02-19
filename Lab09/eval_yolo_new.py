from darkflow.net.build import TFNet
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

threshold = 0.5

options = {"pbLoad": "/home/0613316/darkflow-master/darkflow-master/built_graph/tiny-yolo-lab9.pb",
            "threshold": threshold,
           "metaLoad": "/home/0613316/darkflow-master/darkflow-master/built_graph/tiny-yolo-lab9.meta"
           }

eval_path = 'test_img/'

tfnet = TFNet(options)

def boxing(original_img, predictions, thre):
    newImage = np.copy(original_img)

    for i in len(predictions):
        top_x = predictions[i]['topleft']['x']
        top_y = predictions[i]['topleft']['y']

        btm_x = predictions[i]['bottomright']['x']
        btm_y = predictions[i]['bottomright']['y']

        inner = False

        for j in len(predictions):
            if i != j:
                top_x_2 = predictions[j]['topleft']['x']
                top_y_2 = predictions[j]['topleft']['y']

                btm_x_2 = predictions[j]['bottomright']['x']
                btm_y_2 = predictions[j]['bottomright']['y']
                if top_x>top_x_2 and top_y>top_y_2 and btm_x<btm_x_2 and btm_y<btm_y_2:
                    inner = True
                    break
                    
        if inner == False:
            confidence = predictions[i]['confidence']
            label = predictions[i]['label'] + " " + str(round(confidence, 2))

            if confidence > thre:
                newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_SIMPLEX ,1, (0, 255, 0), 1, cv2.LINE_AA)
                newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)


    return newImage

# Read the video from specified path
# cam = cv2.VideoCapture("C:\\Users\\Admin\\PycharmProjects\\project_1\\openCV.mp4")
videoCapture = cv2.VideoCapture('demo.wmv')
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

videoWriter = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
# frame

while(True):

    ret,frame = videoCapture.read()
    if ret:
        result = tfnet.return_predict(frame)
        if len(result)!=0:
            frame = boxing(frame, result, threshold)
        else:
            print("Nothing detected")

        videoWriter.write(frame)

    else:
        break

# Release all space and windows once done
videoCapture.release()
videoWriter.release()
cv2.destroyAllWindows()
