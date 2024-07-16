import cv2
import numpy as np
import math
 
def thresholding(img):
    blurred_image = cv2.GaussianBlur(img, (3, 3), 0)
    imgHsv = cv2.cvtColor(blurred_image,cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([94,179,0])
    upperWhite = np.array([112,255,144])
    maskWhite = cv2.inRange(imgHsv,lowerWhite,upperWhite)
    return maskWhite
 
def getHistogram(img,minPer=0.1,display=False,region=1):

    if region == 1:
        histValues = np.sum(img, axis=0)
    else:
        histValues = np.sum(img[img.shape[0]//region:,:], axis=0)

    maxValue = np.max(histValues)
    minValue = minPer*maxValue

    indexArray = np.where(histValues >= minValue)
    panjangArray = len(indexArray[0])
    basePoint = int(np.average(indexArray))

    if display:
        imgHist = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
        for x,intensity in enumerate(histValues):
            cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0]-intensity//255//region),(255,0,255,1))
            cv2.circle(imgHist,(basePoint,img.shape[0]),10,(0,255,255),cv2.FILLED)
        return basePoint,imgHist
    
    return basePoint, panjangArray

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getAngle(x,frame_width,frame_height):

    # Calculate the difference in x-coordinates
    delta_x = x - frame_width // 2

    # Calculate the tangent of the angle
    tan_theta = delta_x / frame_height

    # Calculate the angle in radians
    angle_degrees = math.degrees(math.atan(tan_theta))

    return angle_degrees

def rotateImage(image, angle):
    # Dapatkan dimensi gambar dan tentukan titik pusat
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Dapatkan matriks rotasi
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Terapkan rotasi pada gambar
    rotated_image = cv2.warpAffine(image, M, (w, h))

    return rotated_image