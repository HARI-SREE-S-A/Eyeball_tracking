import mediapipe as mp
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mpdraw = mp.solutions.drawing_utils
measurements = mpdraw.DrawingSpec(thickness=1,circle_radius=1)

while True:
    success, img = cap.read()
    h, w, c = img.shape
    left_eye_landmarks = [249, 263, 466, 388, 387, 386]
    right_eye_landmarks = [7, 33, 246, 161, 160, 159]
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpdraw.draw_landmarks(img,facelms,mp_face_mesh.FACEMESH_CONTOURS,measurements,measurements)




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

    threshold = 3 #modify as per need
    print(left_ear, right_ear)

    if left_ear < threshold and right_ear < threshold:
        cv2.putText(img,"open",(10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

    else:
        cv2.putText(img ,"close",(10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    cv2.imshow("harvis", img)
    cv2.waitKey(1)

