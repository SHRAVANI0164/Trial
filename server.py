import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from HandTrackingModule import handDetector
from reportlab.pdfgen import canvas
import io

# Initialize FastAPI
app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
numModel = tf.keras.models.load_model("number_emnist_model.h5")
DIGIT_IMG_SIZE = 28

# Settings
brushThickness = 15
eraserThickness = 100
drawColor = (255, 0, 255)  # Default pink
header = None

# Initialize
detector = handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load header images
header_path = "Header"
overlayList = [cv2.imread(os.path.join(header_path, f)) for f in os.listdir(header_path)]
header = cv2.resize(overlayList[0], (1280, 125))

# --- Helper Functions ---

def crop_canvas(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img, (0, 0, img.shape[1], img.shape[0])
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]
    return cropped, (x, y, w, h)

def preprocess_digit(digit):
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

def segment_digits(processed, expected_width=DIGIT_IMG_SIZE, wide_thresh=1.2):
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > wide_thresh * expected_width:
            num_digits = int(round(w / expected_width))
            num_digits = max(2, num_digits)
            seg_width = w // num_digits
            for i in range(num_digits):
                boxes.append((x + i * seg_width, y, seg_width, h))
        else:
            boxes.append((x, y, w, h))
    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes

def merge_boxes(boxes, gap_threshold=5):
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

def recognize_canvas(canvas_img):
    cropped_img, _ = crop_canvas(canvas_img)
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    boxes = segment_digits(processed)
    merged_boxes = merge_boxes(boxes)

    recognized_digits = ""
    for (x, y, w, h) in merged_boxes:
        digit_roi = processed[y:y + h, x:x + w]
        processed_digit = preprocess_digit(digit_roi)
        processed_digit = processed_digit.astype("float32") / 255.0
        processed_digit = 1.0 - processed_digit
        processed_digit = np.expand_dims(processed_digit, axis=-1)
        processed_digit = np.expand_dims(processed_digit, axis=0)

        prediction = numModel.predict(processed_digit)
        predicted_label = np.argmax(prediction, axis=1)[0]
        recognized_digits += str(predicted_label)

    return recognized_digits

def create_pdf(text, output_path="recognized_output.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    c = canvas.Canvas(output_path)
    c.setFont("Helvetica", 20)
    c.drawString(100, 750, f"Recognized Digits: {text}")
    c.save()
    return output_path
def generate_canvas_stream():
    global xp, yp, imgCanvas, drawColor, header

    while True:
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if lmList:
            x1, y1 = lmList[8][1:3]  # Index finger tip
            x2, y2 = lmList[12][1:3]  # Middle finger tip
            fingers = detector.fingersUp()

            if fingers[1] and fingers[2]:  # TWO fingers up (selection mode)
                xp, yp = 0, 0  # Reset previous points

                # Check if in the header region (top part)
                if y1 < 125:
                    if 0 < x1 < 320:
                        drawColor = (0, 0, 255)  # Red
                    elif 320 < x1 < 640:
                        drawColor = (0, 255, 0)  # Green
                    elif 640 < x1 < 960:
                        drawColor = (255, 0, 0)  # Blue
                    elif 960 < x1 < 1280:
                        drawColor = (0, 0, 0)  # Eraser (black)
            elif fingers[1] and not fingers[2]:  # Only index finger up (Drawing Mode)
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                if drawColor == (0, 0, 0):
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0

        # Combine header, video, and canvas
        img[0:125, 0:1280] = header
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # Encode combined frame
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



# --- FastAPI Endpoints ---

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_canvas_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/capture-and-recognize")
async def capture_and_recognize():
    recognized_text = recognize_canvas(imgCanvas)
    return {"recognized": recognized_text}

@app.post("/save-pdf")
async def save_pdf(background_tasks: BackgroundTasks, text: str):
    pdf_path = create_pdf(text)
    background_tasks.add_task(lambda: None)  # Dummy background task
    return FileResponse(pdf_path, filename="recognized_output.pdf", media_type="application/pdf")


    

# --- Start Server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
