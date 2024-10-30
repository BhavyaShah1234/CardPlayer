import cv2 as cv

cv.namedWindow('FRAME')
cap = cv.VideoCapture(0)

a = ['A', 'B', 'C', 'D', 'E']

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv.imshow('FRAME', frame)
    key = cv.waitKey(1)
    print(key)
    if key == 27:
        break
cap.release()
