import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

actions = [
    "hello", "thank", "please", "yes", "no", "help", "good", "bad", "more", "stop",
    "sorry", "name", "want", "like", "love", "eat", "drink", "water", "food", "friend",
    "family", "home", "work", "school", "time", "day", "night", "today", "now", "where",
    "who", "what", "when", "why", "how", "fast", "slow", "big", "small", "hot",
    "cold", "happy", "sad", "open", "close", "read", "write", "learn", "computer", "understand"
]

DATA_DIR = "./asl_dataset"
sequence_length = 90
target_instances = 40

for action in actions:
    os.makedirs(os.path.join(DATA_DIR, action), exist_ok=True)

model_task_path = "hand_landmarker.task"
if not os.path.exists(model_task_path):
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_task_path
    )

base_options = python.BaseOptions(model_asset_path=model_task_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.75
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print("\n[INFO] Data collector active. Press SPACE to record, TAB to switch word, Q to quit.\n")

current_action_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    current_word = actions[current_action_idx]
    action_dir = os.path.join(DATA_DIR, current_word)
    current_count = len([f for f in os.listdir(action_dir) if f.endswith(".npy")])

    cv2.putText(frame, f"Word [{current_action_idx + 1}/50]: {current_word.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
    cv2.putText(frame, f"Instances: {current_count}/{target_instances}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 197, 94) if current_count >= target_instances else (234, 179, 8), 2)
    cv2.putText(frame, "SPACE: Record | TAB: Next Word | Q: Quit", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)

    cv2.imshow("ASL Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == 9:
        current_action_idx = (current_action_idx + 1) % len(actions)
    elif key == ord(' '):
        print(f"[INFO] Recording sequence for {current_word} ({current_count + 1}/{target_instances})")
        window_data = []

        for _ in range(sequence_length):
            ret, rec_frame = cap.read()
            if not ret:
                break
            rec_frame = cv2.flip(rec_frame, 1)
            image_rgb = cv2.cvtColor(rec_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = detector.detect(mp_image)

            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
                wrist = landmarks[0:3]
                normalized = landmarks - np.tile(wrist, 21)
                window_data.append(normalized)
            else:
                window_data.append(np.zeros(63))

            cv2.putText(rec_frame, "RECORDING...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (34, 197, 94), 2)
            cv2.imshow("ASL Collector", rec_frame)
            cv2.waitKey(30)

        if len(window_data) == sequence_length:
            file_name = f"{int(time.time())}.npy"
            np.save(os.path.join(action_dir, file_name), np.array(window_data))
            print(f"[SUCCESS] Saved to {action_dir}/{file_name}")

cap.release()
cv2.destroyAllWindows()
