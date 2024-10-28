import pandas as pd
import json
import random

dataset = pd.read_csv('music_data.csv')


positive_songs = []
negative_songs = []


def preprocess_dataset(data):
    result = {
        "positive": [],
        "negative": []
    }

    for index, row in data.iterrows():
        title = row['Name']
        artist = row['Artist']
        url = f"https://www.youtube.com/results?search_query={title.replace(' ', '+')}+{artist.replace(' ', '+')}"
        
        
        sentiment = 'positive' if random.choice([True, False]) else 'negative'
        
        song = {
            "title": f"{title} by {artist}",
            "url": url
        }
        
        if sentiment == 'positive':
            result["positive"].append(song)
        else:
            result["negative"].append(song)
    
    return result


preprocessed_data = preprocess_dataset(dataset)


with open('songs.json', 'w') as f:
    json.dump(preprocessed_data, f, indent=2)

print("Preprocessing complete. Data saved to 'songs.json'.")
