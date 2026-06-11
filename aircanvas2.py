import cv2
import numpy as np
import mediapipe as mp

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0

# Colors (BGR)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]  # Blue, Green, Red, Eraser
color_names = ["Blue", "Green", "Red", "Eraser"]
current_color = colors[0]

def fingers_up(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)

    # Other fingers
    for tip in tips:
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)

    return fingers


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw color selection bar
    for i, col in enumerate(colors):
        cv2.rectangle(img, (i*100, 0), ((i+1)*100, 80), col, -1)
        cv2.putText(img, color_names[i], (i*100+10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            h, w, _ = img.shape
            lm = hand_landmarks.landmark

            x1, y1 = int(lm[8].x * w), int(lm[8].y * h)   # index
            x2, y2 = int(lm[12].x * w), int(lm[12].y * h) # middle

            fingers = fingers_up(hand_landmarks)

            # 🖐️ Erase mode (all fingers up)
            if all(fingers):
                current_color = (0, 0, 0)
                cv2.putText(img, "ERASE MODE", (200,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            # ✌️ Selection mode
            elif fingers[1] and fingers[2]:
                prev_x, prev_y = 0, 0

                if y1 < 80:  # top bar
                    for i in range(len(colors)):
                        if i*100 < x1 < (i+1)*100:
                            current_color = colors[i]

            # ☝️ Drawing mode
            elif fingers[1] and not fingers[2]:

                # Brush thickness based on distance
                distance = np.hypot(x2 - x1, y2 - y1)
                thickness = int(distance / 5)
                thickness = max(5, min(thickness, 30))

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1

                # Eraser bigger brush
                if current_color == (0, 0, 0):
                    thickness = 50

                cv2.line(canvas, (prev_x, prev_y), (x1, y1), current_color, thickness)
                prev_x, prev_y = x1, y1

            else:
                prev_x, prev_y = 0, 0

    # Merge canvas
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("AirCanvas", img)

    key = cv2.waitKey(1)

    # 💾 Save
    if key == ord('s'):
        cv2.imwrite("drawing.png", canvas)
        print("Saved as drawing.png")

    # 🧽 Clear
    if key == ord('c'):
        canvas = np.zeros_like(img)

    # ❌ Exit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
