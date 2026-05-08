import pandas as pd
import pickle
import os
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train():
    print("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("parasharmanas/movie-recommendation-system")
    
    movies_path = os.path.join(path, 'movies.csv')
    ratings_path = os.path.join(path, 'ratings.csv')

    print("Loading data...")
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    print("Processing popularity...")
    # Calculate number of ratings for each movie
    movie_popularity = ratings.groupby('movieId').size().reset_index(name='rating_count')
    
    # Merge with movies
    movies = movies.merge(movie_popularity, on='movieId', how='left')
    movies['rating_count'] = movies['rating_count'].fillna(0)

    # Filter top 10,000 movies to avoid OOM issues and keep recommendations relevant
    print("Filtering top 10,000 movies...")
    movies = movies.sort_values(by='rating_count', ascending=False).head(10000).reset_index(drop=True)

    # Preprocess genres
    print("Preprocessing genres...")
    movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ', regex=False)
    movies['combined'] = movies['genres']

    print("Vectorizing and calculating similarity...")
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(movies['combined'])
    # Use float32 to save space and memory
    similarity = cosine_similarity(matrix).astype('float32')

    print("Saving model...")
    with open('model.pkl', 'wb') as f:
        pickle.dump((movies, similarity), f)
    print(f"Model saved as model.pkl with {len(movies)} movies.")

if __name__ == '__main__':
    train()
