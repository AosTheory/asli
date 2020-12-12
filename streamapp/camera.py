from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
face_detection_videocam = cv2.CascadeClassifier(os.path.join(
			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
face_detection_webcam = cv2.CascadeClassifier(os.path.join(
			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
# load our serialized face detector model from disk
prototxtPath = os.path.sep.join([settings.BASE_DIR, "face_detector/deploy.prototxt"])
weightsPath = os.path.sep.join([settings.BASE_DIR,"face_detector/res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(os.path.join(settings.BASE_DIR,'face_detector/mask_detector.model'))

backSub = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=15, detectShadows=False)

class IPWebCam(object):
	def __init__(self):
		self.url = "http://10.0.0.240:8080/shot.jpg"

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
		blur = cv2.GaussianBlur(gray, (7, 7), 2)
		canny = cv2.Canny(blur, 30, 30)
		fgMask = backSub.apply(canny)

		faces_detected = face_detection_webcam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
		for (x, y, w, h) in faces_detected:
			face = fgMask[y:y + h, x:x + w]
			fgMask[y:y + h, x:x + w] = np.zeros_like(face)



		resize = cv2.resize(img, (1280, 720), interpolation = cv2.INTER_LINEAR) 		
		frame_flip = cv2.flip(resize,1)
		# font 
		font = cv2.FONT_HERSHEY_SIMPLEX 
		# org 
		org = (50, 650) 
		# fontScale 
		fontScale = 2
		#Color in BGR 
		color = (255, 255, 255) 
		# Line thickness of 2 px 
		thickness = 4 
		sign_text = cv2.putText(frame_flip, 'Sample ASL Interpreted Text', org, font,  
						fontScale, color, thickness, cv2.LINE_AA) 
		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()


