# CineMatch | Premium Movie Recommender

CineMatch is an AI-powered movie recommendation system that uses Content-Based Filtering and Natural Language Processing (NLP) to suggest movies based on user favorites.

![CineMatch Screenshot](https://raw.githubusercontent.com/maneomkar369/movie-recommender/master/screenshot.png) *(Note: Add a screenshot later)*

## 🚀 Features
- **AI Recommendations**: Uses TF-IDF Vectorization and Cosine Similarity to find similar movies.
- **Premium UI**: Modern, responsive glassmorphism design with a dark mode aesthetic.
- **Large Dataset**: Trained on 10,000 top-rated movies.
- **Optimized**: High-performance similarity engine with compressed model storage.

## 🛠️ Tech Stack
- **Backend**: Python, Flask
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), JavaScript
- **ML/Data**: Pandas, Scikit-learn, Kagglehub
- **Deployment**: Render

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/maneomkar369/movie-recommender.git
   cd movie-recommender
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**:
   ```bash
   python train_model.py
   ```

5. **Run the app**:
   ```bash
   python app.py
   ```

## 🌐 Deployment
This project is configured for **Render**. It uses the `render.yaml` blueprint to automatically:
- Download datasets via `kagglehub`.
- Train the model during the build process.
- Serve the app using `gunicorn`.

## 📄 License
MIT License
