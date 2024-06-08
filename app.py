from flask import Flask, render_template, Response
import cv2
import handtrackingmodule as htm
import numpy as np

app = Flask(__name__)

cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionCon=0.85)
brushThickness = 15
eraserThickness = 100
xp, yp = 0, 0
imgCanvas = None

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global xp, yp, imgCanvas
    if imgCanvas is None:
        imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            img = cv2.flip(img, 1)
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]
                fingers = detector.fingersUp()

                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0
                    if y1 < 125:
                        if 250 < x1 < 450:
                            drawColor = (255, 0, 255)
                        elif 550 < x1 < 750:
                            drawColor = (255, 0, 0)
                        elif 800 < x1 < 950:
                            drawColor = (0, 255, 0)
                        elif 1050 < x1 < 1200:
                            drawColor = (0, 0, 0)
                    cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

                if fingers[1] and not fingers[2]:
                    cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    if drawColor == (0, 0, 0):
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    else:
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                    xp, yp = x1, y1

            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, imgCanvas)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)