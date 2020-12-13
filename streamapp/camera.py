from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
from streamapp.boundingBox import *
from tensorflow.python.keras.models import load_model, model_from_json
from streamapp.preprocessing import *

face_detection_videocam = cv2.CascadeClassifier(os.path.join(
			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
face_detection_webcam = cv2.CascadeClassifier(os.path.join(
			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
# load our serialized face detector model from disk
prototxtPath = os.path.sep.join([settings.BASE_DIR, "models/deploy.prototxt"])
weightsPath = os.path.sep.join([settings.BASE_DIR,"models/res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(os.path.join(settings.BASE_DIR,'models/mask_detector.model'))

backSub = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=15, detectShadows=False)

letter_map = {1:"A",2:"B",3:"C",4:"D",5:"del",6:"E",7:"F",8:"G",9:"H",10:"I",11:"J",12:"K",13:"L",14:"M",15:"N",16:"nothing",17:"O",18:"P",
              19:"Q",20:"R",21:"S",22:"space",23:"T",24:"U",25:"V",26:"W",27:"X",28:"Y",29:"Z"}

class IPWebCam(object):
	def __init__(self):
		self.url = "http://10.0.0.240:8080/shot.jpg"
		self.counter = 1 
		model_path = "models/v3-ft10-model.json"
		weights_path = "models/InceptionV3_weights_safe.01-0.17.h5"
		self.sign_output = ''


		# with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
		with open(model_path, 'r') as file:
			self.model = model_from_json(file.read())
			self.model.load_weights(weights_path) 		

	def __del__(self):
		cv2.destroyAllWindows()

	def get_frame(self):
		imgResp = urllib.request.urlopen(self.url)
		imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
		img = cv2.imdecode(imgNp,-1)
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (5, 5), 0)
		canny = cv2.Canny(blur, 50, 200, 3, L2gradient=True)
		fgMask = backSub.apply(canny)

		faces_detected = face_detection_webcam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
		for (x, y, w, h) in faces_detected:
			face = fgMask[y:y + h, x:x + w]
			fgMask[y:y + h, x:x + w] = np.zeros_like(face)

		resize = cv2.resize(img, (1280, 720), interpolation = cv2.INTER_LINEAR) 	
		fgMask = cv2.resize(fgMask, (1280, 720), interpolation = cv2.INTER_LINEAR) 	
		frame_flip = cv2.flip(resize,1)
		# font 
		font = cv2.FONT_HERSHEY_SIMPLEX 
		# org 
		org = (50, 700) 
		# fontScale 
		fontScale = 2
		#Color in BGR 
		color = (255, 255, 255) 
		# Line thickness of 2 px 
		thickness = 4 
		# cv2.putText(frame_flip, 'Sample ASL Interpreted Text', org, font,  
		# 				fontScale, color, thickness, cv2.LINE_AA)
		left, right, top, bottom = initBoundingBox(frame_flip)
		imgBox = drawBoundingBox(frame_flip, left, right, top, bottom)
		fgMask_flip = cv2.flip(fgMask,1)
		boundingBox = getBoxAsImage(fgMask_flip, left, right, top, bottom)
		if self.counter > 30:
			x = boundingBox
			# cv2.imwrite('boundedHand.jpg', boundingBox)
			# x = image.img_to_array(boundingBox)
			print(x.shape)
			# boundingBox = np.tile(boundingBox,(1,1,3))
			x = scale(x, 200, 200)
			x = np.expand_dims(x, axis=2)
			print(x.shape)
			x = np.repeat(x, 3, axis=2)
			print(x.shape)
			# x = edge_filter(boundingBox)
			
			y = self.model.predict(np.expand_dims(x/255.0, axis=0))
			print(y[0])
			y = np.argmax(y[0])
			self.sign_output += letter_map[y+1]
			print(self.sign_output)
			self.counter = 0
		self.counter += 1
		cv2.putText(imgBox, self.sign_output, org, font,  
				fontScale, color, thickness, cv2.LINE_AA)
		ret, jpeg = cv2.imencode('.jpg', imgBox)
		return jpeg.tobytes()


