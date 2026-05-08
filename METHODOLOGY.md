# Methodology: CineMatch Recommendation System

This document outlines the technical approach and mathematical foundations used in the CineMatch movie recommender.

## 1. Overview
CineMatch uses a **Content-Based Filtering** approach. This method recommends items similar to those a user liked in the past by analyzing the internal attributes (features) of the items—in this case, movie genres and metadata.

## 2. Data Pipeline

### A. Data Sourcing
The system utilizes the MovieLens dataset, a standard benchmark in recommendation systems research. Data is retrieved programmatically using `kagglehub` to ensure the latest metadata is available.

### B. Popularity Pruning
To ensure high-quality recommendations and system stability, the dataset is filtered to include only the top 5,000 movies based on the number of user ratings. This removes "noise" from obscure titles and ensures the similarity matrix fits within cloud memory constraints (512MB RAM).

## 3. Feature Engineering

### A. TF-IDF Vectorization
We transform text-based genres into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The weight of a term $t$ in document $d$ is calculated as:
$$W_{t,d} = TF_{t,d} \times \log\left(\frac{N}{DF_t}\right)$$

Where:
- **TF**: Frequency of the genre in a specific movie.
- **N**: Total number of movies.
- **DF**: Number of movies containing that genre.

This ensures that unique genre combinations (e.g., "Sci-Fi Horror") create a stronger "signature" than common ones.

## 4. Recommendation Engine

### A. Cosine Similarity
To measure the similarity between two movies, we calculate the cosine of the angle between their TF-IDF vectors ($A$ and $B$):

$$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

- **1.0**: Perfectly similar.
- **0.0**: No similarity.

### B. Pre-computation
The system computes the entire 5,000 x 5,000 similarity matrix during the deployment build phase. This allows the web server to return recommendations in **sub-millisecond time** without performing heavy calculations on every user request.

## 5. System Architecture
- **Web Framework**: Flask (Python)
- **Production Server**: Gunicorn (WSGI)
- **ML Libraries**: Scikit-learn, Pandas, Numpy
- **Cloud Infrastructure**: Render (PaaS)
