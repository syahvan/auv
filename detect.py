import cv2
import cvzone
from ultralytics import YOLO
import math
from sort import *
import numpy as np
import utlis
import time
from datetime import datetime
import os

# Load YOLO model for object detection
model = YOLO("model/auv_openvino_model", task="detect")
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

def getPipeCurve(img, imgOri, imageDetect, display=2):
    """
    Calculate the steering angle based on the detected base point.

    Parameters:
    img (numpy.ndarray): Input image.
    imgOri (numpy.ndarray): Original input image.
    imageDetect (numpy.ndarray): Image after detection.
    display (int): Display mode (0: none, 1: result, 2: stacked).

    Returns:
    angle (int): Calculated steering angle.
    END (bool): Flag indicating if the end condition is met.
    """
    imgResult = img.copy()
    imageDetect = imageDetect.copy()

    # Thresholding
    imgThres = utlis.thresholding(img)

    # Finding BasePoint
    basePoint, lenIndex = utlis.getHistogram(imgThres, display=False, minPer=0.5)
    if lenIndex > 420:
        END = 1
    else:
        END = 0

    # Finding Angle
    angleRaw = utlis.getAngle(basePoint, frame_width, frame_height)
    angleList.append(angleRaw)
    if len(angleList) > avgVal:
        angleList.pop(0)
    angle = int(sum(angleList) / len(angleList))

    # Display Output
    if display != 0:
        cv2.line(imgResult, (frame_width // 2, frame_height), (frame_width // 2, frame_height // 2), (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(imgResult, (frame_width // 2, frame_height), (basePoint, frame_height // 2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.line(imgResult, (frame_width // 2, frame_height // 2), (basePoint, frame_height // 2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cvzone.putTextRect(imgResult, f'Steering Angle: {str(angle)}', (0, 30), scale=1.3, thickness=2, offset=3)
    if display == 2:
        imgStacked = utlis.stackImages(0.9, ([imgOri, imgResult], [imgThres, imageDetect]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Result', imgResult)
    
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
    """
    imageDetect = img.copy()

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

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if (currentClass == "Bocor" or currentClass == "Retak") and conf > 0.5:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(imageDetect, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        if currentClass == "Bocor":
            color = bocorColor
        elif currentClass == "Retak":
            color = retakColor

        cvzone.cornerRect(imageDetect, (x1, y1, w, h), l=9, rt=2, colorR=color, colorC=color)
        cvzone.putTextRect(imageDetect, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                           scale=1.1, thickness=2, offset=3, colorR=color)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(imageDetect, (cx, cy), 3, color, cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1] + 10:
            now = datetime.now()
            timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
            if bocorCount.count(id) == 0 and currentClass == "Bocor":
                bocorCount.append(id)
                cv2.line(imageDetect, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)
                filename = os.path.join('hasil', f"bocor_{timestamp}.jpg")
                cv2.imwrite(filename, imageDetect)
                print(f"Frame saved: {filename}")
            elif retakCount.count(id) == 0 and currentClass == "Retak":
                retakCount.append(id)
                cv2.line(imageDetect, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)
                filename = os.path.join('hasil', f"retak_{timestamp}.jpg")
                cv2.imwrite(filename, imageDetect)
                print(f"Frame saved: {filename}")

    totalBocor = len(bocorCount)
    totalRetak = len(retakCount)

    cvzone.putTextRect(imageDetect, f'Total Bocor: {totalBocor}', (0, 30), scale=1.3, thickness=2, offset=3)
    cvzone.putTextRect(imageDetect, f'Total Retak: {totalRetak}', (0, 55), scale=1.3, thickness=2, offset=3)

    return imageDetect, totalBocor, totalRetak

def main():
    """
    Main function to run the pipeline for damage detection and steering angle calculation.

    Capture video frames, perform damage detection, calculate the steering angle,
    and display the results with FPS and average FPS.
    """
    cap = cv2.VideoCapture('test.mp4')
    cap.set(3, frame_width)
    cap.set(4, frame_height)
    
    # Variables to calculate FPS
    frame_count = 0
    start_time = time.time()
    fps = 0
    average_fps = 0

    writer = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (frame_width, frame_height))

    # Tracking
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    while True:
        # Press 'q' to stop
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
        
        img = cv2.resize(img, (480, 480))
        writer.write(img)
        imgOri = img.copy()

        # Increment frame count
        frame_count += 1

        # Calculate FPS
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time

        # Calculate Average FPS
        average_fps = (average_fps * (frame_count - 1) + fps) / frame_count

        # Display the resulting frame
        cvzone.putTextRect(imgOri, f'FPS: {fps:.2f}', (0, 30), scale=1.3, thickness=2, offset=3)
        cvzone.putTextRect(imgOri, f'Avg FPS: {average_fps:.2f}', (0, 55), scale=1.3, thickness=2, offset=3)
        
        imageDetect, totalBocor, totalRetak = detectDamage(model, img, tracker)
        angle, END = getPipeCurve(img, imgOri, imageDetect)

        print(f'Angle: {angle}, Total Bocor: {totalBocor}, Total Retak: {totalRetak}')

        if END:
            # Release resources
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

        serial = f"{angle};{END};{totalBocor};{totalRetak}"

if __name__ == '__main__':
    main()
