import cv2
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input("Enter gesture label (play / next / mute etc): ")
samples_required = 200

data = []

print("Show your gesture to camera...")

while len(data) < samples_required:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in handLms.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row.append(label)
            data.append(row)

    cv2.putText(frame, f"Samples: {len(data)}/{samples_required}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Collecting Gestures", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)

if os.path.exists("gestures.csv"):
    df.to_csv("gestures.csv", mode="a", header=False, index=False)
else:
    df.to_csv("gestures.csv", index=False)

print("Gesture saved successfully ✅")