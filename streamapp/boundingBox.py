# Bounding box functionality

from skimage.measure import compare_ssim
import cv2
import numpy as np

backSub = cv2.createBackgroundSubtractorMOG2(history=60,
                                             varThreshold=15,
                                             detectShadows=False)

# Runs edge detection then background subtraction on an image
def preProcMask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 2)
    canny = cv2.Canny(blur, 30, 30)
    fgMask = backSub.apply(canny)
    return fgMask


# initializes and returns left, right, top and bottom of bounding box
def initBoundingBox(frame):
    h, w = frame.shape[:2]
    left = int(w/2)
    right = int(w*9/10)
    top = int(h/8)
    bottom = int(h*7/8)
    return left, right, top, bottom


# Returns the bounding box cut from the image
def getBoxAsImage(frame, left, right, top, bottom):
    newFrame = frame.copy()
    return newFrame[top:bottom, left:right]

# Resizes the frame to a new width and height
def resizeImage(frame, width, height):
    newDim = (width, height)
    return cv2.resize(frame.copy(), newDim, interpolation=cv2.INTER_AREA)

# Draws bounding box and text on copy of original frame
def drawBoundingBox(frame, left, right, top, bottom):
    newFrame = frame.copy()
    textPos = (left, top-20)
    cv2.putText(newFrame, "Signing Box", textPos, cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 5)
    cv2.rectangle(newFrame, (left, top), (right, bottom), (0, 255, 0), 3)
    return newFrame


# compares two instances of bounding boxes and returns score rep. similarity
# score is in [-1,1] with a score of 1 being perfectly similar, and -1 being inverted images
def compareBoxes(prevBox, postBox):
    prevGray = cv2.cvtColor(prevBox, cv2.COLOR_BGR2GRAY)
    postGray = cv2.cvtColor(postBox, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(prevGray, postGray, full=True)
    return score


# takes in new frame along with the previous bounding box as image
# and the location of the box in the frame. Calculates new box that most closely matches last box
# and returns coords of the new box
def updateBoundingBox(newFrame, lastBox, left, right, top, bottom):
    height, width = newFrame.shape[:2]
    horiDelta = int(abs((right - left) / 10))
    vertDelta = int(abs((bottom - top) / 10))

    # scores : init, up, down, left, right
    scores = np.array([-2, -2, -2, -2, -2])

    initBox = newFrame[top:bottom, left:right]
    scores[0] = compareBoxes(lastBox, initBox)

    upTop = max(top - vertDelta, 0)
    upBottom = max(bottom - vertDelta, 100)
    upBox = newFrame[upTop:upBottom, left:right]
    scores[1] = compareBoxes(lastBox, upBox)

    downTop = min(top + vertDelta, height - 100)
    downBottom = min(bottom + vertDelta, height)
    downBox = newFrame[downTop:downBottom, left:right]
    scores[2] = compareBoxes(lastBox, downBox)

    leftLeft = max(left - horiDelta, 0)
    leftRight = max(right - horiDelta, 100)
    leftBox = newFrame[top:bottom, leftLeft:leftRight]
    scores[3] = compareBoxes(lastBox, leftBox)

    rightLeft = min(left + horiDelta, width - 100)
    rightRight = min(right + horiDelta, width)
    rightBox = newFrame[top:bottom, rightLeft:rightRight]
    scores[4] = compareBoxes(lastBox, rightBox)

    maxInd = np.argmax(scores)

    if maxInd == 1:
        # up box
        return left, right, upTop, upBottom
    elif maxInd == 2:
        # down box
        return left, right, downTop, downBottom
    elif maxInd == 3:
        # left box
        return leftLeft, leftRight, top, bottom
    elif maxInd == 4:
        # right box
        return rightLeft, rightRight, top, bottom
    else:
        # init box will be base case
        return left, right, top, bottom
