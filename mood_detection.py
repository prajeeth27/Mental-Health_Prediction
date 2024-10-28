from fer import FER
import cv2

def detect_mood(frame):
    emotion_detector = FER()
    result = emotion_detector.detect_emotions(frame)

    if result:
        emotions = result[0]['emotions']
        mood = max(emotions, key=emotions.get)
        bounding_box = result[0]['box']
        return mood, bounding_box
    return None, None
def generate_frames():
    cap = cv2.VideoCapture(0)
    emotion_detector = FER()
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        result = emotion_detector.detect_emotions(frame)

        if result:
            emotions = result[0]['emotions']
            mood = max(emotions, key=emotions.get)
            bounding_box = result[0]['box']
            (x, y, w, h) = bounding_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, mood, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

