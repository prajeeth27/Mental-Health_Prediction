import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib


def read_csv_with_encodings(file_path):
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            print(f"Failed to decode with {encoding}. Trying next encoding.")
            continue
    raise ValueError("Unable to decode file with available encodings.")


try:
    train_df = read_csv_with_encodings('train.csv')
except FileNotFoundError:
    print("File 'train.csv' not found. Make sure the file exists in the correct directory.")
    exit(1)
except ValueError as e:
    print(f"Error reading 'train.csv': {e}")
    exit(1)


print(train_df.head())


if 'text' not in train_df.columns or 'sentiment' not in train_df.columns:
    print("Dataset must contain 'text' and 'sentiment' columns.")
    exit(1)


train_df = train_df.dropna(subset=['text', 'sentiment'])


X = train_df['text']
y = train_df['sentiment']


sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
y = y.map(sentiment_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])


model_pipeline.fit(X_train, y_train)


accuracy = model_pipeline.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

joblib.dump(model_pipeline, 'sentiment_model.pkl')
print("Model saved as 'sentiment_model.pkl'.")


joblib.dump(model_pipeline.named_steps['tfidf'], 'tfidf_vectorizer.pkl')
print("TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'.")
