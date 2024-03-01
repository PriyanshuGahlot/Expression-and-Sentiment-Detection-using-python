import cv2
import pathlib

facefile = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

faceDetectionModel = cv2.CascadeClassifier(str(facefile))

camera = cv2.VideoCapture(0)

while(True):
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetectionModel.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
    if(cv2.waitKey(1)==ord("e")):
        scale_factor = 1.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        x_new = max(0, x - (new_width - width) // 2)
        y_new = max(0, y - (new_height - height) // 2)
        cropped_frame = frame[y_new:y_new + new_height, x_new:x_new + new_width]
        cv2.imshow("Cropped Last Frame", cropped_frame)
    for(x,y,width,height) in faces:
        cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2)
    cv2.imshow("Faces",frame)
    if(cv2.waitKey(1)==ord("q")):
        break

camera.release()
cv2.destroyWindow()