import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def load_model():
    return joblib.load('sentiment_model.pkl')

def predict_sentiment(model_pipeline, text):
    text_tfidf = model_pipeline.named_steps['tfidf'].transform([text])
    prediction = model_pipeline.named_steps['clf'].predict(text_tfidf)[0]
    return 'positive' if prediction == 1 else 'negative'
