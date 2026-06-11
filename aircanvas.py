import cv2
import numpy as np
import mediapipe as mp

# Initialize
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Canvas
canvas = None

# Previous points
prev_x, prev_y = 0, 0

cap = cv2.VideoCapture(0)

# ✅ FULLSCREEN SETUP
cv2.namedWindow("Air Drawing", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Air Drawing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    # Convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip (id = 8)
            h, w, _ = img.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # Draw circle on finger
            cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)

            # Draw line
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)

            prev_x, prev_y = x, y

    # Merge canvas and webcam
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    # Instructions
    cv2.putText(img, "Press C to Clear | Q to Quit",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Air Drawing", img)

    # ✅ FIXED KEY HANDLING (IMPORTANT)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        canvas = np.zeros_like(img)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
