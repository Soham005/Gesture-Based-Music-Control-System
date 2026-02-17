import cv2
import mediapipe as mp
import pyautogui
import joblib
import time

# ---------------- SAFETY ---------------- #
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("gesture_model.pkl")

# ---------------- MEDIAPIPE ---------------- #
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- CONFIG ---------------- #
COOLDOWN = 0.8
STABILITY_FRAMES = 5

last_action_time = 0
prediction_buffer = []

cap = cv2.VideoCapture(0)


def trigger_action(gesture):
    try:
        if gesture == "play":
            pyautogui.press("k")

        elif gesture == "next":
            pyautogui.hotkey("shift", "n")

        elif gesture == "prev":
            pyautogui.hotkey("shift", "p")

        elif gesture == "mute":
            pyautogui.press("m")

        elif gesture == "volup":
            pyautogui.press("up")

        elif gesture == "voldown":
            pyautogui.press("down")

    except Exception as e:
        print("Automation error:", e)


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    stable_prediction = None

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            prediction = model.predict([landmarks])[0]

            prediction_buffer.append(prediction)
            if len(prediction_buffer) > STABILITY_FRAMES:
                prediction_buffer.pop(0)

            stable_prediction = max(
                set(prediction_buffer),
                key=prediction_buffer.count
            )

    else:
        prediction_buffer.clear()

    now = time.time()

    if stable_prediction and (now - last_action_time > COOLDOWN):
        trigger_action(stable_prediction)
        last_action_time = now

    if stable_prediction:
        cv2.putText(frame, f"Gesture: {stable_prediction}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
