import mediapipe as mp
import cv2
import numpy as np
import pyautogui
import time


def switch_tab(b):
    b+=1
    windows = pyautogui.getAllTitles()
    window_title = "My Window Title"
    if window_title in windows:
        window = pyautogui.getWindowsWithTitle(window_title)[0]
        if window:
            window = windows[0]
            window.activate()

    pyautogui.keyDown('win')
    pyautogui.press("tab")
    pyautogui.keyUp("win")
    for i in range(0,4):
        time.sleep(0.1)
        pyautogui.press("tab")
        time.sleep(0.2)
    pyautogui.press("enter")


def switchtab_next():
    windows = pyautogui.getAllTitles()
    window_title = "My Window Title"
    if window_title in windows:
        window = pyautogui.getWindowsWithTitle(window_title)[0]
        if window:
            window = windows[0]
            window.activate()

    pyautogui.press("enter")

    print("here im")


font = cv2.FONT_HERSHEY_SIMPLEX
position = (100, 200)
color = (0, 0, 255)
thickness = 2

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mpdraw = mp.solutions.drawing_utils
measurements = mpdraw.DrawingSpec(thickness=1, circle_radius=1)
b = 0

while True:
    success, img = cap.read()
    h, w, c = img.shape
    left_eye_landmarks = [249, 263, 466, 388, 387, 386]
    right_eye_landmarks = [7, 33, 246, 161, 160, 159]
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpdraw.draw_landmarks(img, facelms, mp_face_mesh.FACEMESH_CONTOURS, measurements, measurements)
    else:
        cv2.putText(img, "No face Detected !!", (20, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 10)


    def eye_aspect_ratio(eye):
        A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
        B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
        C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))

        ear = (A + B) / (2.0 * C)
        return ear


    left_eye_pts = [results.multi_face_landmarks[0].landmark[i] for i in left_eye_landmarks]
    right_eye_pts = [results.multi_face_landmarks[0].landmark[i] for i in right_eye_landmarks]

    left_ear = eye_aspect_ratio(left_eye_pts)
    right_ear = eye_aspect_ratio(right_eye_pts)

    threshold = 3
    print(left_ear, right_ear)

    if left_ear < threshold and right_ear < threshold:
        b = 0
        cv2.putText(img, "open", (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)


    else:
        b += 1

        cv2.putText(img, "close", (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        cv2.putText(img, str(b), (20, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        if b > 100:

            cv2.putText(img, "Alert triggered", (100, 250), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 8)
            switch_tab(b)
            # for i in range(10):
            # cv2.putText(img, 'alert', position, font, 1, color, thickness, cv2.LINE_AA)
            # color = (color[1], color[2], color[0])  # change the color
            # position = (position[0] + 5, position[1] - 5)  # change the position
    cv2.imshow("harvis", img)
    cv2.waitKey(1)

