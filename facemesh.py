import cv2
import mediapipe as mp
from math import sqrt

mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh()
mpdraw = mp.solutions.drawing_utils
measurements = mpdraw.DrawingSpec(thickness=1,circle_radius=1)

cap = cv2.VideoCapture(0)


while True:
    border,image = cap.read()

    imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = facemesh.process(imgRGB)

    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpdraw.draw_landmarks(image,facelms,mpfacemesh.FACEMESH_CONTOURS,measurements,measurements)
            for id,lm in enumerate(facelms.landmark):
                h,w,c = image.shape
                cx,cy,cz = int(lm.x*w),int(lm.y*h),int(lm.z*c)
                landmark1 = results.multi_face_landmarks[0].landmark[42]
                landmark2 = results.multi_face_landmarks[0].landmark[36]
                distance = sqrt((landmark2.x - landmark1.x)**2 + (landmark2.y - landmark1.y)**2 + (landmark2.z - landmark1.z)**2)
                distance = (distance*1000000)
                print(distance)
                if distance < 18602:
                    cv2.putText(image,"close",(10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
                else:
                    cv2.putText(image, "open", (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)


                print(id,cx,cy)






    cv2.imshow("harvis",image)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

