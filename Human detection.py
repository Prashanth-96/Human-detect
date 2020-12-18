# #HUMAN AND OBJECT DETECTION USING YOLOV3

import cv2
import numpy as np 



def load_yolo():
 	net = cv2.dnn.readNet('enter the weights path here/yolov3-spp.weights', 'enter the cfg file path here/yolov3-spp.cfg')
 	classes = []
 	with open("enter the coco.names path here", "r") as f:
	   classes = [line.strip() for line in f.readlines()]
 	layers_names = net.getLayerNames()
 	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
 	colors = np.random.uniform(0, 255, size=(len(classes), 3))
 	return net, classes, colors, output_layers

def load_image(img_path):
 	# image loading
 	img = cv2.imread(img_path)
 	img = cv2.resize(img, None, fx=0.4, fy=0.4)
 	height, width, channels = img.shape
 	return img, height, width, channels

def detect_objects(img, net, outputLayers):			
 	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
 	net.setInput(blob)
 	outputs = net.forward(outputLayers)
 	return blob, outputs


def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
# 			print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
 indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
 font = cv2.FONT_HERSHEY_PLAIN
 for i in range(len(boxes)):
  if i in indexes:
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    color = colors[i]
    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imwrite('enter the location where the image to be saved',img)
 cv2.imshow("Image", img)     
 if cv2.waitKey(1) & 0xff==27:
     exit
cv2.destroyAllWindows()
  
def image_detect(img_path): 
 model, classes, colors, output_layers = load_yolo()
 image, height, width, channels = load_image(img_path)
 blob, outputs = detect_objects(image, model, output_layers)
 boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
 draw_labels(boxes, confs, colors, class_ids, classes, image)
 print('press q to exit')
 if cv2.waitKey(0) & 0xFF == ord('q'):
     cv2.destroyAllWindows()
 


def start_video(video_path):
 model, classes, colors, output_layers = load_yolo()
 cap = cv2.VideoCapture(video_path)
 while (cap.isOpened()):
  ret, frame = cap.read()
  height, width, channels = frame.shape
  blob, outputs = detect_objects(frame, model, output_layers)
  boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
  draw_labels(boxes, confs, colors, class_ids, classes, frame)
  if cv2.waitKey(100) & 0xFF == ord('q'): 
     cv2.destroyAllWindows()
     cv2.waitkey(0)
     cap.release()
     cv2.destroyAllWindows()
     cv2.waitkey(0)
     break
 cap.release()
 cv2.destroyAllWindows() 
  


         

       
if __name__=='__main__':
     opt=int(input('Welcome to human detection module,press 1 for detecting human from image\n and 2 for detecting from video'))
     if(opt==1):
         
         print('Task started successfully')
         image_detect('enter the image path here ')
     elif(opt==2):
         print('task started successfully')
         start_video('enter the video path here')
     else:
         print('invalid entry from user')
          




