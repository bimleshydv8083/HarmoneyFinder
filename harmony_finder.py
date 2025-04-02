import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from collections import defaultdict

class HarmonyFinder:
    def __init__(self):
        # Sample music dataset (in a real app, this would come from a database)
        self.songs = pd.DataFrame([
            {"title": "Blinding Lights", "artist": "The Weeknd", "genre": "pop", "mood": "energetic", "tempo": "fast", "popularity": 90},
            {"title": "Levitating", "artist": "Dua Lipa", "genre": "pop", "mood": "happy", "tempo": "medium", "popularity": 85},
            {"title": "Good as Hell", "artist": "Lizzo", "genre": "pop", "mood": "confident", "tempo": "medium", "popularity": 80},
            {"title": "Sunflower", "artist": "Post Malone", "genre": "hip-hop", "mood": "chill", "tempo": "medium", "popularity": 88},
            {"title": "Someone Like You", "artist": "Adele", "genre": "pop", "mood": "sad", "tempo": "slow", "popularity": 92},
            {"title": "Thunderstruck", "artist": "AC/DC", "genre": "rock", "mood": "energetic", "tempo": "fast", "popularity": 87},
            {"title": "Bohemian Rhapsody", "artist": "Queen", "genre": "rock", "mood": "epic", "tempo": "variable", "popularity": 95},
            {"title": "Blinding Lights", "artist": "The Weeknd", "genre": "pop", "mood": "energetic", "tempo": "fast", "popularity": 90},
            {"title": "Shape of You", "artist": "Ed Sheeran", "genre": "pop", "mood": "happy", "tempo": "medium", "popularity": 89},
            {"title": "Circles", "artist": "Post Malone", "genre": "hip-hop", "mood": "chill", "tempo": "medium", "popularity": 84}
        ])
        
        # User preferences storage
        self.user_preferences = defaultdict(dict)
        
        # Create feature vectors for songs
        self.songs['features'] = self.songs['genre'] + ' ' + self.songs['mood'] + ' ' + self.songs['tempo']
        self.vectorizer = TfidfVectorizer()
        self.song_vectors = self.vectorizer.fit_transform(self.songs['features'])
    
    def recommend_by_mood(self, mood, n=5):
        """Recommend songs based on mood"""
        mood_songs = self.songs[self.songs['mood'] == mood]
        if len(mood_songs) >= n:
            return mood_songs.sample(n)
        return mood_songs
    
    def recommend_by_preferences(self, user_id, n=5):
        """Recommend songs based on user's listening history and preferences"""
        if user_id not in self.user_preferences or not self.user_preferences[user_id].get('history'):
            return self.songs.sample(n)
        
        # Get user's recently played songs
        history = self.user_preferences[user_id]['history']
        liked_songs = self.songs[self.songs['title'].isin(history)]
        
        if len(liked_songs) == 0:
            return self.songs.sample(n)
        
        # Calculate similarity between liked songs and all songs
        liked_features = ' '.join(liked_songs['features'].tolist())
        user_vector = self.vectorizer.transform([liked_features])
        similarities = cosine_similarity(user_vector, self.song_vectors).flatten()
        
        # Get top similar songs not already in history
        similar_indices = similarities.argsort()[::-1]
        recommendations = []
        for idx in similar_indices:
            song = self.songs.iloc[idx]
            if song['title'] not in history:
                recommendations.append(song)
            if len(recommendations) >= n:
                break
                
        return pd.DataFrame(recommendations)
    
    def add_to_history(self, user_id, song_title):
        """Add a song to user's listening history"""
        if 'history' not in self.user_preferences[user_id]:
            self.user_preferences[user_id]['history'] = []
        self.user_preferences[user_id]['history'].append(song_title)
    
    def set_user_mood(self, user_id, mood):
        """Set user's current mood"""
        self.user_preferences[user_id]['mood'] = mood

# Example usage
if __name__ == "__main__":
    hf = HarmonyFinder()
    
    # User actions
    user_id = "user123"
    hf.set_user_mood(user_id, "happy")
    hf.add_to_history(user_id, "Blinding Lights")
    hf.add_to_history(user_id, "Levitating")
    
    # Get recommendations
    print("Mood-based recommendations:")
    print(hf.recommend_by_mood("happy"))
    
    print("\nPersonalized recommendations:")
    print(hf.recommend_by_preferences(user_id))