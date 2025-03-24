import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import HandTrackingModule as htm  # Ensure this module is in your project folder
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt

#############################
# Virtual Painter Settings
#############################
brushThickness = 15
eraserThickness = 100

#######################
# Load Header Images
#######################
folderPath = "Header"
if not os.path.exists(folderPath):
    print("Error: The 'Header' folder does not exist!")
    exit()
myList = os.listdir(folderPath)
if len(myList) == 0:
    print("Error: The 'Header' folder is empty.")
    exit()
print("Files in Header folder:", myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(os.path.join(folderPath, imPath))
    if image is None:
        print(f"Error: Could not load {imPath}")
    else:
        overlayList.append(image)
if len(overlayList) == 0:
    print("Error: No valid images loaded from 'Header' folder")
    exit()
header = cv2.resize(overlayList[0], (1280, 125))  # Resize header to fit
drawColor = (255, 0, 255)

#######################
# Webcam Setup
#######################
cap = cv2.VideoCapture(0)  # Use 0 for default camera
cap.set(3, 1280)
cap.set(4, 720)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

###############################################
# Load the digit recognition model (EMNIST digits)
###############################################
numModel = load_model("number_emnist_model.h5")  # Ensure this file is in your working directory
DIGIT_IMG_SIZE = 28  # Model expects 28x28 images


#########################################################
# Function: Preprocess a segmented digit for recognition
#########################################################
def preprocess_digit(digit):
    """
    Resizes the extracted digit while preserving aspect ratio and centers it in a 28x28 canvas.
    Ensures that the new dimensions are at least 1 pixel.
    """
    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(20 * w / h))
    else:
        new_w = 20
        new_h = max(1, int(20 * h / w))
    resized_digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas_digit = np.zeros((DIGIT_IMG_SIZE, DIGIT_IMG_SIZE), dtype=np.uint8)
    x_offset = (DIGIT_IMG_SIZE - new_w) // 2
    y_offset = (DIGIT_IMG_SIZE - new_h) // 2
    canvas_digit[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_digit
    return canvas_digit


#########################################################
# Function: Crop the canvas to drawn area
#########################################################
def crop_canvas(img):
    """Crop the canvas to the bounding box of non-black pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img, (0, 0, img.shape[1], img.shape[0])
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]
    return cropped, (x, y, w, h)


#########################################################
# Function: Segment digits from thresholded image
#########################################################
def segment_digits(processed, expected_width=DIGIT_IMG_SIZE, wide_thresh=1.2):
    """
    Given a thresholded image (white digits on black), find contours.
    For each contour, if the bounding box width is larger than (wide_thresh * expected_width),
    split it into equal segments.
    Returns a list of bounding boxes (x, y, w, h).
    """
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:  # Filter out noise
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # If the box is wide, assume it may contain multiple digits.
        if w > wide_thresh * expected_width:
            num_digits = int(round(w / expected_width))
            num_digits = max(2, num_digits)
            seg_width = w // num_digits
            for i in range(num_digits):
                boxes.append((x + i * seg_width, y, seg_width, h))
        else:
            boxes.append((x, y, w, h))
    # Sort boxes left-to-right
    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes


#########################################################
# Function: Merge nearby bounding boxes
#########################################################
def merge_boxes(boxes, gap_threshold=5):  # Adjusted gap_threshold from 10 to 5
    """
    Merge bounding boxes that are close horizontally.
    boxes: list of (x, y, w, h)
    gap_threshold: maximum gap between boxes to consider them as one.
    """
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    current_box = list(boxes[0])

    for b in boxes[1:]:
        if b[0] - (current_box[0] + current_box[2]) < gap_threshold:
            new_x = current_box[0]
            new_y = min(current_box[1], b[1])
            new_w = (b[0] + b[2]) - current_box[0]
            new_h = max(current_box[3], b[3]) - new_y
            current_box = [new_x, new_y, new_w, new_h]
        else:
            merged.append(tuple(current_box))
            current_box = list(b)
    merged.append(tuple(current_box))
    return merged


#########################################################
# Function: Extract and recognize digits from the canvas
#########################################################
def recognize_canvas(canvas_img, debug=False):
    """
    Processes the drawing canvas by:
      1. Cropping to the drawn area.
      2. Converting to grayscale and thresholding.
      3. Segmenting digits using contours and splitting wide regions.
      4. Preprocessing each digit and predicting with numModel.
    Returns the recognized number as a string.
    """
    cropped_img, _ = crop_canvas(canvas_img)
    if debug:
        cv2.imshow("Cropped Canvas", cropped_img)

    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    # Changed threshold value from 40 to 45
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)

    # Use morphological closing to fill gaps; kernel size reduced to (2, 2)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    boxes = segment_digits(processed, expected_width=DIGIT_IMG_SIZE, wide_thresh=1.2)
    print(f"[DEBUG] Original segmented boxes: {len(boxes)}")

    merged_boxes = merge_boxes(boxes, gap_threshold=5)
    print(f"[DEBUG] Merged boxes: {len(merged_boxes)}")

    if debug:
        debug_img = cropped_img.copy()
        for (x, y, w, h) in merged_boxes:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Debug - Digit Regions", debug_img)

    recognized_digits = ""
    for (x, y, w, h) in merged_boxes:
        digit_roi = processed[y:y + h, x:x + w]
        processed_digit = preprocess_digit(digit_roi)
        processed_digit = processed_digit.astype("float32") / 255.0

        # Invert the digit image if your model expects white digit on black background
        processed_digit = 1.0 - processed_digit

        processed_digit = np.expand_dims(processed_digit, axis=-1)  # (28,28,1)
        processed_digit = np.expand_dims(processed_digit, axis=0)  # (1,28,28,1)

        if debug:
            cv2.imshow("Processed Digit", processed_digit[0, :, :, 0])
            cv2.waitKey(50)

        prediction = numModel.predict(processed_digit)
        predicted_label = np.argmax(prediction, axis=1)[0]
        print(f"Box at x:{x} predicted: {predicted_label}")
        recognized_digits += str(predicted_label)

    return recognized_digits


#########################################################
# Function: Create a PDF with recognized text
#########################################################
def create_pdf(text, output_pdf_path="recognized_output.pdf"):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 20)
    c.drawString(50, height - 50, f"Recognized Text: {text}")
    c.showPage()
    c.save()
    print(f"PDF generated: {output_pdf_path}")


#########################################################
# Main Loop: Virtual Painter with Recognition Integration
#########################################################
while True:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    # Drawing on the canvas
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:3]
        x2, y2 = lmList[12][1:3]
        fingers = detector.fingersUp()

        # Selection Mode: Two fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # Drawing Mode: Index finger up only
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1
        else:
            xp, yp = 0, 0

    # Combine canvas with webcam image
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    img[0:125, 0:1280] = header
    img = cv2.bitwise_or(img, imgCanvas)

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        recognized = recognize_canvas(imgCanvas, debug=True)
        print("Recognized Number:", recognized)
        create_pdf(recognized)
    if key == ord('q'):
        break
    elif key == ord('c') or key == ord('C'):
        imgCanvas = np.zeros((720, 1280, 3), np.uint8)
        print("[INFO] Canvas cleared.")

cap.release()
cv2.destroyAllWindows()
