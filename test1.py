import cv2 as cv2
print(cv2.__version__)

cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 1
defaultPos = [10, 10]
while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

    c = cv2.waitKey(1)
    if c == 99:
        #draw_label(frame, "Captured", defaultPos)
        img = cv2.resize(frame, (224, 224))
        cv2.putText(img,"Hello World!!!", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.putText(frame, "Hello", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imwrite("frame%d.jpg" % count, img)
        print(img.shape)
        count += count

    cv2.imshow('Input', frame)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

