import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize drawing utils for hand landmarks
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        break

    # Convert the image to RGB (MediaPipe works with RGB images)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and get the hand landmarks
    results = hands.process(img_rgb)

    # If hands are detected, draw landmarks
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the image with hand landmarks
    cv2.imshow("Image", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
