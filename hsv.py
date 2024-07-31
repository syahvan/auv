import cv2
import numpy as np

# Baca gambar
img = cv2.imread("HSV.png")

def thresholding(img):
    blurred_image = cv2.GaussianBlur(img, (3, 3), 0)
    imgHsv = cv2.cvtColor(blurred_image,cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([93,179,0])
    upperWhite = np.array([112,255,150])
    maskWhite = cv2.inRange(imgHsv,lowerWhite,upperWhite)
    return maskWhite

result = thresholding(img)

cv2.imwrite("HSV 4.jpg", result)