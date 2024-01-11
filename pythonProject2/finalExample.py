import cv2


recongzer = cv2.face.LBPHFaceRecognizer()
faceXml = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    if not cap.isOpened():
        print("Please open the cap")
        break
    ret, face = cap.read()
    if not ret:
        print("Can't read frame")
        break

    face = cv2.resize(face, (540, 320))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    detecteds = faceXml.detectMultiScale(gray)

    for detected in detecteds:
        x, y, w, h = detected
        cv2.rectangle(face, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Stupid', face)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()