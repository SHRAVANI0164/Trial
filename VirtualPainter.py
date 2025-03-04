import cv2
import numpy as np
import os
import HandTrackingModule as htm

folderPath = "Header"

# Check if the folder exists
if not os.path.exists(folderPath):
    print("Error: The 'Header' folder does not exist!")
    exit()

myList = os.listdir(folderPath)

if len(myList) == 0:
    print("Error: The 'Header' folder is empty.")
    exit()

print("Files in Header folder:", myList)

# Load images from the Header folder
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    if image is None:
        print(f"Error: Could not load {imPath}")
    else:
        overlayList.append(image)

# Ensure at least one image is loaded
if len(overlayList) == 0:
    print("Error: No valid images loaded from 'Header' folder")
    exit()

header = cv2.resize(overlayList[0], (1280, 125))  # Resize to fit

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

detector = htm.handDetector(detectionCon=0.85)

while True:
    # 1. Import Image
    success, img = cap.read()

    if not success:
        print("Error: Failed to capture image")
        continue


    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)

    if len(lmList)!=0:
        print(lmList)

        #tip of index and middle fingers
        x1,y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]



    # 3. Check which fingers are up
    img[0:125, 0:1280] = header

    # 4. If Selection Mode - Two finger are up
    # 5. If Drawing Mode - Index Finger is up
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit the loop

cap.release()
cv2.destroyAllWindows()
