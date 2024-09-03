import cv2
import numpy as np
import math

def thresholding(img):
    """
    Apply Gaussian blur and HSV thresholding to an image.
    
    Parameters:
    img (numpy.ndarray): Input image in BGR format.
    
    Returns:
    maskWhite (numpy.ndarray): Binary mask where white regions are detected.
    """
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Convert the blurred image to HSV color space
    imgHsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for white color
    lowerWhite = np.array([90, 130, 0])
    upperWhite = np.array([112, 255, 150])
    
    # Create a mask for the white color range
    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
    
    return maskWhite

def getHistogram(img, minPer=0.1, display=False, region=1):
    """
    Compute the histogram of the image and find the base point.

    Parameters:
    img (numpy.ndarray): Binary input image.
    minPer (float): Minimum percentage threshold to consider a pixel for the histogram.
    display (bool): Flag to display the histogram image.
    region (int): Region of the image to consider for the histogram.

    Returns:
    basePoint (int): The average base point of the histogram.
    imgHist (numpy.ndarray): Image of the histogram if display is True.
    """
    # Compute the sum of pixel values along the columns
    if region == 1:
        histValues = np.sum(img, axis=0)
    else:
        histValues = np.sum(img[img.shape[0]//region:, :], axis=0)

    # Determine the maximum value in the histogram
    maxValue = np.max(histValues)
    
    # Determine the minimum value to consider based on minPer
    minValue = minPer * maxValue

    # Get the indices where histogram values are above the minimum value
    indexArray = np.where(histValues >= minValue)
    lenIndex = len(indexArray[0])
    
    # Calculate the base point as the average of the indices
    basePoint = int(np.average(indexArray))

    if display:
        # Create an image to display the histogram
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histValues):
            cv2.line(imgHist, (x, img.shape[0]), (x, img.shape[0] - intensity // 255 // region), (255, 0, 255, 1))
        cv2.circle(imgHist, (basePoint, img.shape[0]), 10, (0, 255, 255), cv2.FILLED)
        return basePoint, imgHist, lenIndex
    
    return basePoint, lenIndex

def stackImages(scale, imgArray):
    """
    Stack multiple images into a single image.

    Parameters:
    scale (float): Scaling factor for the images.
    imgArray (list): List of images to be stacked.

    Returns:
    ver (numpy.ndarray): Stacked image.
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    
    return ver

def getAngle(x, frame_width, frame_height):
    """
    Calculate the angle between a point and the center of the image.

    Parameters:
    x (int): x-coordinate of the point.
    frame_width (int): Width of the image frame.
    frame_height (int): Height of the image frame.

    Returns:
    angle_degrees (float): Angle in degrees between the point and the center.
    """
    # Calculate the difference in x-coordinates
    delta_x = x - frame_width // 2

    # Calculate the tangent of the angle
    tan_theta = delta_x / (frame_height // 2)

    # Calculate the angle in degrees
    angle_degrees = math.degrees(math.atan(tan_theta))

    return angle_degrees
