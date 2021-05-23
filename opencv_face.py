#Read a video stream from web cam
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
 

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_color,1.3,5)
        yes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
           cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
        cv2.imshow("Video Frame",frame)
        cv2.imshow("Gray Frame",gray_frame)

    #Wait for user input

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
            