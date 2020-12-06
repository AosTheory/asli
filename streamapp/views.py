from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from streamapp.camera import IPWebCam
import cv2
import numpy as np
# Create your views here.


def index(request):
	return render(request, 'streamapp/home.html')


def generate(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def webcam_feed(request):
	return StreamingHttpResponse(generate(IPWebCam()),
					content_type='multipart/x-mixed-replace; boundary=frame')
