# Methodology: CineMatch Movie Recommendation System

## 1. Project Overview
CineMatch is a content-based movie recommendation engine that suggests films similar to a user’s selected title. The system leverages TF-IDF vectorization and cosine similarity to compute genre-based similarities, deployed as a lightweight web application. The primary goal is to deliver fast, explainable, and high-quality recommendations without requiring user interaction history.

## 2. Research Objectives
- Build a recommendation system using only movie metadata (genres).
- Achieve sub-second response times under cloud resource constraints.
- Handle cold-start scenarios (new movies or new users) effectively.
- Provide a simple, interactive web interface for real-time suggestions.

## 3. Theoretical Foundation

### 3.1 Content-Based Filtering Paradigm
Content-based filtering recommends items by comparing a target item’s features against all other items in the catalog. Unlike collaborative filtering, it does not require user rating data. The underlying assumption is: if a user likes a particular movie, they will also like movies with similar attributes.

### 3.2 Text Representation – TF-IDF
Movies are described by a list of genres (e.g., ["Action", "Sci-Fi", "Thriller"]). To convert this categorical text into a numerical feature vector, we apply **Term Frequency-Inverse Document Frequency (TF-IDF)**.

- **Term Frequency (TF)**: Measures how often a genre appears in a movie’s genre list. Since each genre typically appears once (binary), TF simplifies to 1 if present, 0 otherwise.
- **Inverse Document Frequency (IDF)**: Reduces the weight of genres that are very common across all movies (e.g., “Drama”) and increases the weight of rare, discriminative genres (e.g., “Film-Noir”).

The TF-IDF weight for genre $t$ in movie $d$ is:
$$w_{t,d} = TF_{t,d} \times \log\left(\frac{N}{DF_t}\right)$$

**Where:**
- $N$ = Total number of movies in the dataset.
- $DF_t$ = Number of movies that contain genre $t$.

### 3.3 Similarity Measure – Cosine Similarity
Given two movie vectors $A$ and $B$ (derived from TF-IDF), the cosine similarity measures the angle between them:

$$\text{similarity}(A,B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

- Value ranges from **0** (no shared genres) to **1** (identical genre profiles).
- Cosine similarity is preferred over Euclidean distance because it is robust to vector magnitude variations – only the direction (genre presence pattern) matters.

## 4. Implementation Methodology

### 4.1 Data Acquisition and Preprocessing
- **Dataset**: MovieLens latest small dataset (via `kagglehub`) containing:
    - `movies.csv`: Movie IDs, titles, and genres (pipe-separated).
    - `ratings.csv`: User ratings (used for popularity pruning).
- **Popularity Pruning**: Retain only the top 5,000 movies with the highest number of user ratings to eliminate obscure titles and stay within memory limits (≤512 MB).
- **Genre Cleaning**: Split the genres field on the pipe (|) character and handle missing entries.

### 4.2 Feature Vectorization
Using scikit-learn’s `TfidfVectorizer`:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
```
Output is a sparse matrix of shape (5000, unique_genres).

### 4.3 Similarity Matrix Pre-computation
All pairwise cosine similarities are computed once during the build phase:
```python
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype('float32')
```
`cosine_sim` is a dense 5000 × 5000 matrix stored in memory. At runtime, the engine simply looks up pre-computed values.

### 4.4 Recommendation Logic
1. Find the movie’s index in the dataset.
2. Retrieve the corresponding row from `cosine_sim`.
3. Sort similarity scores in descending order.
4. Exclude the input movie itself.
5. Return the top N movies (default N = 5).

## 5. Software Architecture
| Layer | Technology / Tool |
| :--- | :--- |
| **Backend** | Python 3.12 |
| **Web Framework** | Flask |
| **ML Libraries** | scikit-learn, pandas, numpy |
| **Production** | Gunicorn (WSGI server) |
| **Frontend** | HTML, CSS (Glassmorphism), JS |
| **Cloud Hosting** | Render (PaaS, 512 MB RAM plan) |

## 6. Performance Optimization
- **Pre-computation**: All heavy numeric work is done during build.
- **Memory Management**: Similarity matrix stored as `float32` to save 50% memory.
- **Complexity**: Runtime lookup is $O(k \log k)$ for sorting 5,000 scores, which is negligible.

## 7. Testing Methodology
| Test Case | Description | Expected Outcome |
| :--- | :--- | :--- |
| **Exact match** | Input "Toy Story" | Returns similar animated/family films |
| **Partial match** | Input "toy story" | Case-insensitive match works |
| **Memory footprint** | Monitor RAM on Render | Stays below 400 MB |
| **Response time** | Measure latency | < 100 ms per request |

## 8. Results and Evaluation
- **Accuracy**: Correctly identifies genre hybrids (e.g., Action/Sci-Fi).
- **Cold-start**: New movies with genres can be recommended immediately.
- **Efficiency**: Average 27ms per query on Render Free Tier.

## 9. Future Enhancements
- **Hybrid Approach**: Combine with Collaborative Filtering (SVD).
- **Feature Expansion**: Include cast, director, and plot summaries.
- **Caching**: Implement Redis for high-traffic deployments.

## 10. Conclusion
CineMatch demonstrates a production-ready recommendation system using classical techniques. By pre-computing the similarity matrix and optimizing for memory, it achieves high speed and reliability on constrained cloud environments.