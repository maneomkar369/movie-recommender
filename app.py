from flask import Flask, request, render_template
import pickle
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def load_model():
    try:
        if os.path.exists('model.pkl'):
            print("Loading existing model...")
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading model.pkl: {e}. Attempting fallback...")
    
    # Fallback: train on the fly if model.pkl doesn't exist or is incompatible
    print("Model not found or incompatible, training on the fly...")
    if not os.path.exists('data/movies.csv'):
        print("Data file not found. Please run train_model.py first.")
        return None, None
    
    try:
        movies = pd.read_csv('data/movies.csv')
        movies['genres'] = movies['genres'].fillna('')
        movies['overview'] = movies['overview'].fillna('')
        movies['combined'] = movies['genres'] + " " + movies['overview']
        
        tfidf = TfidfVectorizer(stop_words='english')
        matrix = tfidf.fit_transform(movies['combined'])
        similarity = cosine_similarity(matrix).astype('float32')
        
        # Try to save the fallback model
        try:
            with open('model.pkl', 'wb') as f:
                pickle.dump((movies, similarity), f)
        except:
            pass
            
        return movies, similarity
    except Exception as e:
        print(f"Fallback training failed: {e}")
        return None, None

movies, similarity = load_model()

def recommend(title, n=5):
    if movies is None or similarity is None:
        return []
        
    # Find movie index (case-insensitive partial match)
    # We use a more specific match if possible
    query = title.lower().strip()
    matches = movies[movies['title'].str.lower().str.contains(query)]
    
    if matches.empty:
        return []
    
    # Prioritize exact matches if they exist
    exact_match = matches[matches['title'].str.lower() == query]
    if not exact_match.empty:
        idx = exact_match.index[0]
    else:
        idx = matches.index[0]
        
    scores = list(enumerate(similarity[idx]))
    # Sort by similarity score, skip the first one (itself)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    recommendations = []
    for i in scores:
        rec_movie = movies.iloc[i[0]]
        # Handle cases where 'overview' might not exist in the dataframe
        overview = rec_movie.get('overview', 'No overview available for this movie.')
        if pd.isna(overview):
            overview = 'No overview available.'
            
        recommendations.append({
            'title': rec_movie['title'],
            'genres': rec_movie['genres'],
            'overview': overview
        })
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        movie_query = request.form.get('movie', '').strip()
        if not movie_query:
            return render_template('index.html', recs=None)
            
        recs = recommend(movie_query)
        return render_template('index.html', recs=recs, movie=movie_query)
        
    return render_template('index.html', recs=None)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
