import numpy as np
from cv2 import cv2
import matplotlib as plt
import os

""" 
1) Detect the bounding box of the hand
2) Find the contours of the hand
3) Background Subtract
4) Edge Detection (may omit)

1) Find joints from joint_detection
2) Other skeletal information?
"""

SHOW = False

#-------------------------------------------
# Returns contours of the input 
# https://github.com/SAint7579/Hand-Detection-with-contours/blob/master/Detection.py
#-------------------------------------------
def contour(frame): 
    # Remove face using cascades
    def blackout(frame,gray):
        face_cascade = cv2.CascadeClassifier('hand_detection/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray,1.2,1)
        for (x,y,w,h) in faces:
            #Blacking out the face
            frame[y:y+h+50,x:x+w] = 0
        return frame

    frame_wof = np.copy(frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    # Removing the face
    frame_wof = blackout(frame_wof,gray)
    #Converting the color scheme
    hsv = cv2.cvtColor(cv2.medianBlur(frame_wof,15),cv2.COLOR_BGR2HSV)
    
    lower = np.array([0, 10, 60])     #Lower range of HSV color
    upper = np.array([40, 165, 255])  #Upper range of HSV color
    #Creating the mask
    mask = cv2.inRange(hsv,lower,upper)
    #Removing noise form the mask
    mask = cv2.dilate(mask,None,iterations=2)
    
    #Extracting contours from the mask
    cnts,_ = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        for c in cnts:
            #To prevent the detection of the noise
            if cv2.contourArea(c) > 8000:
                #Fixing covex defect
                hull = cv2.convexHull(c)
                #Drawing the contours
                cv2.drawContours(frame,[hull],0,(0,0,255),2)
                #Creating the bounding rectangel
                x,y,w,h = cv2.boundingRect(hull)
                cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),(0,255,0),2)
    #Showing the mask
    cv2.imshow("Image",frame)
    cv2.waitKey(0)
    cv2.imshow("Mask",mask)
    cv2.waitKey(0)

#-------------------------------------------
# Find the convex hull that encapsulates the hand
#-------------------------------------------
def convex_hull(frame):
    pass

#-------------------------------------------
# https://gogul.dev/software/hand-gesture-recognition-
# To segment the region of the hand in the image
# Doesnt work when the background is brighter than hand
#-------------------------------------------
def segment(image, grayimage, threshold=75):
    # threshold the image to get the foreground which is the hand
    thresholded = cv2.threshold(grayimage, threshold, 255, cv2.THRESH_BINARY)[1]
    print("Original image shape - " + str(image.shape))
    print("Gray image shape - " + str(grayimage.shape))

    # show the thresholded image
    cv2.imshow("Thesholded", thresholded)

    # get the contours in the thresholded image (_, cnts, _)
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # analyze the contours
        print("Number of Contours found = " + str(len(cnts))) 
        cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
        if SHOW:
            cv2.imshow('All Contours', image) 
            cv2.waitKey(0)
        
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        cv2.drawContours(image, segmented, -1, (0, 255, 0), 3)
        if SHOW:
            cv2.imshow('Max Contour', image) 
            cv2.waitKey(0)
        
        return (thresholded, segmented)

#-------------------------------------------------------------------
# Reduces the resolution of the original image to dims given in args
#-------------------------------------------------------------------
def scale(frame, output_height, output_width):
    # store image shape
    (h, w) = frame.shape[:2]

    #scale_percent = 40 # percent of original size
    #width = int(img.shape[1] * scale_percent / 100) 
    #height = int(img.shape[0] * scale_percent / 100) 
    dim = (output_width, output_height) 
    scaled = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) 
    if SHOW:
        cv2.imshow('resized image', scaled)
        cv2.waitKey(0)
    return scaled

#---------------------------------------------------------------------------
# Performs Background Subtraction using moving averages over multiple frames
#---------------------------------------------------------------------------
def background_subtraction_dynamic(frames):
    pass

#---------------------------------------------------------------------------
# Performs Background Subtraction for a single frame by finding the bounding
# box of the hand and subtracting everything around it
#---------------------------------------------------------------------------
def background_subtraction_static(frame):
    pass

#---------------------------------------------------------------------------
# Returns the edges of the hand after background subtraction
#---------------------------------------------------------------------------
def edge_filter(frame):
    # print(frame.shape)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(frame, 50, 200, 3, L2gradient=True)
    #print("edge dims: ", str(edges.shape))
    if SHOW:
        cv2.imshow('original', frame)
        cv2.imshow('image', edges)
        cv2.waitKey(0)
    return edges

#---------------------------------------------------------------------------
# Returns whether the input sequence of frames is static or dynamic
# Can be performed by measuring variance or feeding to ConvNet
#---------------------------------------------------------------------------
def is_static(frames):
    pass

#---------------------------------------------------------------------------
# Transforms 3D RGB image to 2D grayscale image
#---------------------------------------------------------------------------
def grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

#---------------------------------------------------------------------------
# Uses above functions to prepare the input to the classifier
#---------------------------------------------------------------------------
def preprocess(frames):
    pass

# Testing purposes a
if __name__ == "__main__":

    directory = "data/archive/asl_alphabet_train"

    #for letter in os.listdir(directory):
    #    os.mkdir(os.path.join("data\\archive\\edge_train", letter))

    for letter in os.listdir(directory):
        print(letter)
        for entry in os.listdir(os.path.join(directory, letter)):
            frame = cv2.imread(os.path.join(directory, letter, entry))
            if SHOW: 
                cv2.imshow('original', frame)
                cv2.waitKey(0)
            edges = edge_filter(frame)
            path = os.path.join("data/archive/edge_train", letter, entry)
            #print(path)
            if not cv2.imwrite(path, edges):
                raise Exception("Could not write image")
            
    
    """ for letter in os.scandir(directory):
        for entry in os.scandir(letter.path):
            edges = edge_filter(frame)
            cv2.imwrite(directory + "/" letter.path, img)
            print(entry.path)    
 """
    #frame = cv2.imread("data/archive/asl_alphabet_train/A/A1000.jpg")
    #frame = scale(frame, 224, 224)
    #print(frame.shape)
    #print(frame.shape)
    #contour(frame)
    #segment(frame, grayscale(frame), threshold=150)
    #edge_filter(frame)


    """plt.figure()
    plt.title(name)
    plt.imsave('images/canny_' + name, edges, cmap='gray', format='png')
    plt.imshow(edges, cmap='gray')
    plt.show()"""