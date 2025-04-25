import cv2
import mediapipe as mp
from pygame import mixer
import numpy as np

# Initialize pygame mixer
mixer.init()
mixer.music.load("music.wav")

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eye aspect ratio calculation
def eye_aspect_ratio(landmarks, eye_indices):
    A = ((landmarks[eye_indices[1]][0] - landmarks[eye_indices[5]][0])**2 +
         (landmarks[eye_indices[1]][1] - landmarks[eye_indices[5]][1])**2)**0.5
    B = ((landmarks[eye_indices[2]][0] - landmarks[eye_indices[4]][0])**2 +
         (landmarks[eye_indices[2]][1] - landmarks[eye_indices[4]][1])**2)**0.5
    C = ((landmarks[eye_indices[0]][0] - landmarks[eye_indices[3]][0])**2 +
         (landmarks[eye_indices[0]][1] - landmarks[eye_indices[3]][1])**2)**0.5
    return (A + B) / (2.0 * C)

# Thresholds
thresh = 0.25
frame_check = 20
flag = 0

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                         for lm in face_landmarks.landmark]

            # Define eye landmark indices (based on Mediapipe's Face Mesh)
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(landmarks, left_eye_indices)
            right_ear = eye_aspect_ratio(landmarks, right_eye_indices)
            ear = (left_ear + right_ear) / 2.0

            # Draw eye contours
            left_eye = [landmarks[i] for i in left_eye_indices]
            right_eye = [landmarks[i] for i in right_eye_indices]
            cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)

            # Check if EAR is below the threshold
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                flag = 0

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()