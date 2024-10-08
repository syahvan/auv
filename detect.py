import cv2
import cvzone
from ultralytics import YOLOv10
import math
from sort import *
import numpy as np
import pandas as pd
import utlis
import time
from datetime import datetime
import os
from pathlib import Path
import serial

# Load YOLO model for object detection
model = YOLOv10("model/auv_openvino_model", task="detect")
# Serial Communication
arduino = serial.Serial('COM12', 9600, timeout=.1)
# Variable
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
DETECT = 0

def getPipeCurve(img):
    """
    Calculate the steering angle based on the detected base point.

    Parameters:
    img (numpy.ndarray): Input image.

    Returns:
    angle (int): Calculated steering angle.
    END (bool): Flag indicating if the end condition is met.
    """
    imgResult = img.copy()

    # Apply thresholding to the image
    imgThres = utlis.thresholding(img)

    # Find the base point and its length index
    basePoint, imgHist, lenIndex = utlis.getHistogram(imgThres, display=True, minPer=0.3)
    if lenIndex < 5:
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
    cv2.line(imgResult, (frame_width // 2, frame_height), (frame_width // 2, frame_height // 2), (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.line(imgResult, (frame_width // 2, frame_height), (basePoint, frame_height // 2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.line(imgResult, (frame_width // 2, frame_height // 2), (basePoint, frame_height // 2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cvzone.putTextRect(imgResult, f'Steering Angle: {str(angle)}', (0, 30), scale=1.3, thickness=2, offset=3)

    print(f'Angle: {angle}')
    
    return angle, imgThres, imgHist, imgResult, END


def detectDamage(model, img, tracker):
    """
    Detect damage (leak or crack) in the image using the YOLO model and track detected objects.

    Parameters:
    model (YOLO): Pre-trained YOLO model.
    img (numpy.ndarray): Input image.
    tracker (Sort): Sort tracker object.

    Returns:
    imageDetect (numpy.ndarray): Image with detected and tracked objects.
    DETECT (int): Indicator for the type of damage detected (0 for none, 1 for leak, 2 for crack).
    detection_data (list): List of tuples containing detection time, fps, currentClass, and confidence.
    """
    imageDetect = img.copy()
    DETECT = 0
    detect_dir = Path("./result/images")
    detect_dir.mkdir(exist_ok=True)

    # Perform detection using the YOLO model
    results = model(imageDetect, stream=True, imgsz=480, verbose=True)

    detections = np.empty((0, 5))
    detection_data = []  # Store detection data

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

                # Capture detection time
                detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                detection_data.append([detection_time, currentClass, conf])

    # Update tracker with new detections
    resultsTracker = tracker.update(detections)

    # Draw the detection limits line
    cv2.line(imageDetect, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)

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
                filename = os.path.join(detect_dir, f"bocor_{timestamp}.jpg")
                cv2.imwrite(filename, imageDetect)
                print(f"Frame saved: {filename}")
                DETECT = 1
            elif retakCount.count(id) == 0 and currentClass == "Retak":
                retakCount.append(id)
                cv2.line(imageDetect, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)
                filename = os.path.join(detect_dir, f"retak_{timestamp}.jpg")
                cv2.imwrite(filename, imageDetect)
                print(f"Frame saved: {filename}")
                DETECT = 1
        else:
            DETECT = 0

    totalBocor = len(bocorCount)
    totalRetak = len(retakCount)

    print(f'Total Bocor: {totalBocor}, Total Retak: {totalRetak}')

    # Display total counts on the image
    cvzone.putTextRect(imageDetect, f'Total Bocor: {totalBocor}', (0, 30), scale=1.3, thickness=2, offset=3)
    cvzone.putTextRect(imageDetect, f'Total Retak: {totalRetak}', (0, 55), scale=1.3, thickness=2, offset=3)

    return imageDetect, detection_data, DETECT

def display_video(fps, imgThres, imgHist, imgAngle, imageDetect=None, mode=0):
    # Write FPS
    cvzone.putTextRect(imgAngle, f'FPS: {fps:.2f}', (0, 55), scale=1.3, thickness=2, offset=3)
    # Stack images for display
    if mode == 1:
        imgStacked = utlis.stackImages(0.9, ([imgThres, imgHist], [imgAngle, imageDetect]))
        cv2.imshow('ImageStack', imgStacked)
    elif mode == 2:
        imgStacked = utlis.stackImages(0.9, ([imgAngle, imageDetect]))
        cv2.imshow('ImageStack', imgStacked)
    elif mode == 3:
        imgStacked = utlis.stackImages(0.9, ([imgAngle, imgThres]))
        cv2.imshow('ImageStack', imgStacked)
    elif mode == 4:
        cv2.imshow('Angle', imgAngle)
    elif mode == 5:
        cvzone.putTextRect(imageDetect, f'FPS: {fps:.2f}', (0, 80), scale=1.3, thickness=2, offset=3)
        cv2.imshow('Detection', imageDetect)

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

    # Initialize dataframe
    detection_df = pd.DataFrame(columns=["time", "class", "confidence", "fps", "inference_time"])
    result_dir = Path("./result")
    result_dir.mkdir(exist_ok=True)
    csv_path = result_dir / "detection_log.csv"

    # Open video file for writing the processed frames
    writer = cv2.VideoWriter('result/record.mp4', cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (frame_width, frame_height))

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

        # Timing inference
        start_time = time.perf_counter()
        # Perform damage detection and track objects
        imageDetect, detection_data, DETECT = detectDamage(model, img, tracker)
        end_time = time.perf_counter()  
        inference_time = (end_time - start_time) * 1000
        
        # Calculate the steering angle and check if the end condition is met
        angle, imgThres, imgHist, imgAngle, END = getPipeCurve(img)

        # Display output image
        display_video(fps, imgThres, imgHist, imgAngle, mode=3)

        # Prepare the message to send to the Arduino
        message = f"{angle};{END};{DETECT}\n"
        send_to_arduino(message)  # Send the message via serial communication

        # Update and save the dataframe
        for detection in detection_data:
            detection.append(round(fps, 2))
            detection.append(round(inference_time, 2))
            detection_series = pd.Series(detection, index=detection_df.columns)
            detection_df = pd.concat([detection_df, detection_series.to_frame().T], ignore_index=True)
        detection_df.to_csv(csv_path, index=False)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()