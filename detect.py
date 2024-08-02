import cv2
import cvzone
from ultralytics import YOLOv10
import math
from sort import *
import numpy as np
import utlis
import time
from datetime import datetime
import os
from pathlib import Path
import serial

# Load YOLO model for object detection
model = YOLOv10("model/auv_openvino_model", task="detect")
# Serial Communication
arduino = serial.Serial('COM11', 9600, timeout=1)
# Variables
classNames = ["Bocor", "Retak"]
frame_width = 480
frame_height = 480
angleList = []
avgVal = 10
limits = [0, frame_height // 2 + 20, frame_width, frame_height // 2 + 20]
bocorCount = []
retakCount = []
bocorColor = (168, 182, 33)
retakColor = (84, 163, 214)

def getPipeCurve(img, fps, imageDetect, display=2):
    """
    Calculate the steering angle based on the detected base point.

    Parameters:
    img (numpy.ndarray): Input image.
    fps (float): Frames per second.
    imageDetect (numpy.ndarray): Image after detection.
    display (int): Display mode (0: none, 1: result, 2: stacked).

    Returns:
    angle (int): Calculated steering angle.
    END (bool): Flag indicating if the end condition is met.
    """
    imgResult = img.copy()
    imageDetect = imageDetect.copy()

    # Apply thresholding to the image
    imgThres = utlis.thresholding(img)

    # Find the base point and its length index
    basePoint, imgHist, lenIndex = utlis.getHistogram(imgThres, display=True, minPer=0.5)
    if lenIndex > 400:
        END = 1
    else:
        END = 0

    # Calculate the raw angle based on the base point
    angleRaw = utlis.getAngle(basePoint, frame_width, frame_height)
    angleList.append(angleRaw)
    if len(angleList) > avgVal:
        angleList.pop(0)
    angle = int(sum(angleList) / len(angleList))

    # Display the output
    if display != 0:
        cv2.line(imgResult, (frame_width // 2, frame_height), (frame_width // 2, frame_height // 2), (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(imgResult, (frame_width // 2, frame_height), (basePoint, frame_height // 2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.line(imgResult, (frame_width // 2, frame_height // 2), (basePoint, frame_height // 2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cvzone.putTextRect(imgResult, f'FPS: {fps:.2f}', (0, 30), scale=1.3, thickness=2, offset=3)
        cvzone.putTextRect(imgResult, f'Steering Angle: {str(angle)}', (0, 55), scale=1.3, thickness=2, offset=3)

    # Stack images for display
    if display == 2:
        imgStacked = utlis.stackImages(0.9, ([imgThres, imgHist], [imgResult, imageDetect]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        imgStacked = utlis.stackImages(0.9, ([imgResult, imageDetect]))
        cv2.imshow('ImageStack', imgStacked)
    
    return angle, END


def detectDamage(model, img, tracker):
    """
    Detect damage (leak or crack) in the image using the YOLO model and track detected objects.

    Parameters:
    model (YOLO): Pre-trained YOLO model.
    img (numpy.ndarray): Input image.
    tracker (Sort): Sort tracker object.

    Returns:
    imageDetect (numpy.ndarray): Image with detected and tracked objects.
    totalBocor (int): Total count of leaks detected.
    totalRetak (int): Total count of cracks detected.
    DETECT (int): Indicator for the type of damage detected (0 for none, 1 for leak, 2 for crack).
    """
    imageDetect = img.copy()
    DETECT = 0

    # Perform detection using the YOLO model
    results = model(imageDetect, stream=True, imgsz=480, verbose=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Set the color based on the class
            if currentClass == "Bocor":
                color = bocorColor
            elif currentClass == "Retak":
                color = retakColor

            # Draw bounding box and class label on the image
            cvzone.cornerRect(imageDetect, (x1, y1, w, h), l=9, rt=2, colorR=color, colorC=color)
            cvzone.putTextRect(imageDetect, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                               scale=1.1, thickness=2, offset=3, colorR=color)
            cv2.circle(imageDetect, (cx, cy), 3, color, cv2.FILLED)

            # Store the detection if it meets the criteria
            if (currentClass == "Bocor" or currentClass == "Retak") and conf > 0.5:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update tracker with new detections
    resultsTracker = tracker.update(detections)

    # Draw the detection limits line
    cv2.line(imageDetect, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)

    result_dir = Path("./hasil")
    result_dir.mkdir(exist_ok=True)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Check if the detected object crosses the limits line
        if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1] + 10:
            now = datetime.now()
            timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
            if bocorCount.count(id) == 0 and currentClass == "Bocor":
                bocorCount.append(id)
                cv2.line(imageDetect, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)
                filename = os.path.join('hasil', f"bocor_{timestamp}.jpg")
                cv2.imwrite(filename, imageDetect)
                print(f"Frame saved: {filename}")
                DETECT = 1
            elif retakCount.count(id) == 0 and currentClass == "Retak":
                retakCount.append(id)
                cv2.line(imageDetect, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)
                filename = os.path.join('hasil', f"retak_{timestamp}.jpg")
                cv2.imwrite(filename, imageDetect)
                print(f"Frame saved: {filename}")
                DETECT = 2
        else:
            DETECT = 0

    totalBocor = len(bocorCount)
    totalRetak = len(retakCount)

    # Display total counts on the image
    cvzone.putTextRect(imageDetect, f'Total Bocor: {totalBocor}', (0, 30), scale=1.3, thickness=2, offset=3)
    cvzone.putTextRect(imageDetect, f'Total Retak: {totalRetak}', (0, 55), scale=1.3, thickness=2, offset=3)

    return imageDetect, totalBocor, totalRetak, DETECT

def send_to_arduino(data):
    """
    Send data to the Arduino via serial communication.

    Parameters:
    data (str): Data to be sent to the Arduino. This will be encoded to UTF-8 before transmission.

    Returns:
    None
    """
    # Encode the data to UTF-8 and send it through the serial port
    arduino.write(data.encode('utf-8'))
    
    # Print a confirmation message indicating the sent data
    print(f"Serial Sent: {data}")



def main():

    # Open video file for reading
    cap = cv2.VideoCapture(0)
    cap.set(3, frame_width)  # Set the frame width
    cap.set(4, frame_height) # Set the frame height
    
    # Variables to calculate FPS (frames per second)
    prev_frame_time = 0
    new_frame_time = 0

    # Open video file for writing the processed frames
    writer = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (frame_width, frame_height))

    # Initialize object tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    while cap.isOpened():
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
        
        try:
            # Read a frame from the video
            success, img = cap.read()
            if not success:
                break
        except Exception as e:
            print(e)
            continue
        
        # Resize the image to the required dimensions
        img = cv2.resize(img, (480, 480))
        writer.write(img)  # Write the frame to the video file

        # Calculate FPS
        new_frame_time = time.time() 
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time 

        # Perform damage detection and track objects
        imageDetect, totalBocor, totalRetak, DETECT = detectDamage(model, img, tracker)
        
        # Calculate the steering angle and check if the end condition is met
        angle, END = getPipeCurve(img, fps, imageDetect, display=1)

        print(f'Angle: {angle}, Total Bocor: {totalBocor}, Total Retak: {totalRetak}')

        if END:
            # Release resources when end condition is met
            writer.release()

        # Prepare the message to send to the Arduino
        message = f"{angle};{END};{DETECT}"
        send_to_arduino(message)  # Send the message via serial communication

if __name__ == '__main__':
    main()