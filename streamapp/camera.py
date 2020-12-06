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
			# cv2.rectangle(fgMask, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
			face = fgMask[y:y + h, x:x + w]
			fgMask[y:y + h, x:x + w] = np.zeros_like(face)

		
	
		# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		# lower_skin_hsv = np.array([0, 48, 80], dtype=np.uint8)
		# upper_skin_hsv = np.array([20, 255, 255], dtype=np.uint8)
		# skin_region_hsv = cv2.inRange(hsv_img, lower_skin_hsv, upper_skin_hsv)
		# blurred = cv2.blur(skin_region_hsv, (2,2))
		# ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
		
		# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		#cv2.drawContours(img, [contours], -1, (255,255,0), 2)

		# hull = cv2.convexHull(contours)
		# cv.drawContours(img, [hull], -1, (0, 255, 255), 2)

		# hull = cv.convexHull(contours, returnPoints=False)
		# defects = cv.convexityDefects(contours, hull)
		resize = cv2.resize(fgMask, (640, 480), interpolation = cv2.INTER_LINEAR) 
		frame_flip = cv2.flip(resize,1)
		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()


