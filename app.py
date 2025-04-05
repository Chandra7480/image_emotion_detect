from flask import Flask, render_template, Response
import cv2
import joblib

app = Flask(__name__)

# Load model and face detector
svm_model = joblib.load("emotion_detection_model.pkl")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (48, 48)).flatten().reshape(1, -1)
                emotion_pred = svm_model.predict(face_resized)[0]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                cv2.putText(frame, emotion_pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
