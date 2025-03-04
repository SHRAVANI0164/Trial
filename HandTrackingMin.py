import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize drawing utils for hand landmarks
mp_drawing = mp.solutions.drawing_utils


pTime = 0
cTime = 0

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
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx , cy  = int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                # if id ==0:
                    cv2.circle(img, (cx,cy),25,(255,0,255),cv2.FILLED)

            mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Show the image with hand landmarks
    cv2.imshow("Image", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
