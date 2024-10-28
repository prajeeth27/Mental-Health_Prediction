from flask import Flask, Response, render_template, jsonify, request
import cv2
from fer import FER
import joblib
import random
import json
import pandas as pd


app = Flask(__name__)



try:
    model_pipeline = joblib.load('sentiment_model.pkl')
    preprocessor = joblib.load('data_preprocessor.pkl')
    model = joblib.load('new_stress_prediction_model.pkl')
except FileNotFoundError:
    print("Model file 'sentiment_model.pkl' not found.")
    exit(1)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)




with open('songs.json') as f:
    songs = json.load(f)


motivational_responses = [
    "Every day may not be good, but there's something good in every day.",
    "Keep going, you're doing great!",
    "It's just a bad day, not a bad life.",
    "You are stronger than you think.",
    "Believe in yourself and all that you are.",
    "Challenges are what make life interesting.",
    "Stay positive and work hard."
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    text = request.form.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        text_tfidf = model_pipeline.named_steps['tfidf'].transform([text])
        prediction = model_pipeline.named_steps['clf'].predict(text_tfidf)[0]
    except Exception as e:
        return jsonify({"error": "Error during prediction"}), 500

    sentiment = 'positive' if prediction == 1 else 'negative'
    response = ""
    song = ""

    if sentiment == 'negative':
        response = random.choice(motivational_responses)
        song = random.choice(songs['negative'])
    else:
        song = random.choice(songs['positive'])

    return jsonify(sentiment=sentiment, response=response, song=song)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def load_songs():
    with open('songs.json', 'r') as file:
        return json.load(file)


@app.route('/playlist', methods=['GET'])
def get_playlist():
    try:
        
        data = load_songs()
        
        
        all_songs = data['positive'] + data['negative']
        
        
        selected_songs = random.sample(all_songs, min(10, len(all_songs)))
        
        return jsonify(selected_songs)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    input_data = pd.DataFrame([form_data])
    processed_data = preprocessor.transform(input_data)
    prediction = model.predict(processed_data)
    prediction_text = "High Stress" if prediction[0] == 1 else "Low Stress"
    return render_template('result.html', prediction=prediction_text)
if __name__ == '__main__':
    app.run(debug=True)
