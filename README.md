# Music2Movie: Personalized Recommendation System from Sound to Screen

## Introduction

In today's digital age, personalized content is crucial for user engagement, and recommendation systems are essential for enhancing entertainment experiences. While current platforms excel at suggesting content within a single medium like movies or music, cross-domain recommendations remain largely unexplored. This project, **Music2Movie**, explores the exciting intersection of music and movies. It aims to develop a system that recommends movies based on the mood and vibe of a user's favorite songs, bridging the gap between auditory and visual entertainment. By leveraging artificial intelligence and deep learning, Music2Movie seeks to redefine content discovery, offering a dynamic and emotionally resonant journey from sound to screen.

## Prediction, Inference, and Goals

The primary goal of Music2Movie is to build a predictive system that outputs a numerical match score (between 0 and 1) representing the compatibility between a song and a movie. The system takes audio features from songs (valence, energy, tempo, genre from Spotify Tracks dataset) and movie attributes (sentiment from descriptions, genre, runtime, popularity from TMDb Movies dataset) as inputs. The prediction task focuses on evaluating the alignment of mood, energy, and stylistic features between songs and movies. Inference goals include understanding cross-domain relationships, such as the link between musical energy and action genres, and the impact of mood positivity on compatibility. Ultimately, the project aims to create a scalable framework that enhances cross-media content discovery and informs personalization strategies for entertainment platforms.

## Dataset Description

This project utilizes two main datasets:

### Spotify Dataset

*   **Source:** Hugging Face ([1] Maharshipandya. (2023). Spotify Tracks Dataset. Hugging Face. Retrieved from [https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset))
*   **Description:** Contains audio features and metadata for songs on Spotify, including danceability, energy, mood, and loudness. Each row represents a unique track with attributes defining its rhythmic, tonal, and structural qualities.
*   **Number of Rows and Columns:**
    *   Rows (Tracks): 113,999
    *   Columns (Features): 20

### TMDb Movies Dataset

*   **Source:** Kaggle ([2] Asaniczka. (2023). TMDb Movies Dataset 2023 (930K+ Movies). Kaggle. Retrieved from [https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies))
*   **Description:** Provides detailed metadata for over 930,000 movies, capturing attributes like genre, production companies, and popularity. This dataset is used to align movies with music characteristics for meaningful recommendations.
*   **Number of Rows and Columns:**
    *   Rows (Movies): 1,110,815
    *   Columns (Features): 24

## Preliminary Data Exploration

### Spotify Dataset EDA

Exploratory Data Analysis (EDA) for the Spotify dataset involved:

*   Descriptive statistics to summarize key metrics.
*   Histograms and density plots for visualizing the distribution of popularity and tempo.
*   Box plots comparing the impact of explicit vs. non-explicit content on popularity.
*   Bar charts highlighting the most popular genres.
*   Scatter plots with popularity encoding to reveal relationships between valence and danceability.
*   Correlation heatmap to explore relationships among audio features (e.g., strong correlation between energy and loudness).

Key Findings:

*   **Danceability drives popularity:** Danceability has a stronger impact on track success than valence (emotional positivity), suggesting listeners value rhythm over mood in commercial music.
*   **Niche genres gain popularity:** Genres like anime and grunge show surprisingly high popularity, indicating the influence of niche communities.
*   **External factors matter:** Weak correlations between popularity and audio features highlight the importance of marketing, playlist placement, and promotion in determining a song's success beyond its musical properties.

### TMDb Dataset EDA

EDA for the TMDb dataset focused on feature relevance for recommendation systems:

*   Correlation analysis between numerical features revealed strong relationships between:
    *   Revenue and budget (0.69)
    *   Vote count and revenue (0.7)
*   Exploration of unparsed categorical data (genres, production countries, spoken languages) using Cramér's V correlation showed weak relationships.
*   Word cloud from movie keywords indicated frequent use of explicit or mature terms.

Interesting Comparison:

*   **Explicit Content Discrepancy:**  Only a small portion (8.5%) of Spotify tracks are marked explicit, while TMDb movie keywords often contain mature terms. This difference might impact cross-domain recommendation analysis.

These insights provide a foundation for aligning mood, themes, and genres across music and movie data for developing the recommendation system.

## Methodology

The methodology aims to create a recommendation system that, given a song title and artist, outputs five recommended movie titles and their similarity scores. The key steps include:

*   **Preprocessing of TMDB Dataset:**
    1.  **Duplicate Removal:** Eliminated duplicate rows.
    2.  **Handling Missing Values:** Removed rows with missing values in `title`, `overview`, `poster_path`, `popularity`, and `genres`.
    3.  **Filtering by Release Status:** Retained only movies with `release_status` "Released".
    4.  **Explicit Content Exclusion:** Filtered out movies flagged as explicit (`adult = True`).
    5.  **Column Reduction:** Dropped irrelevant columns like `ids` and `adult`.
    *   Resulted dataset: 436,591 rows and 13 columns (24% of original).

*   **Poster Feature Extraction:**
    *   Utilized movie poster URLs from the TMDB dataset.
    *   Employed a pre-trained MobileNetV2 model (fine-tuned for image classification) to generate feature embeddings from poster images.
    *   Implemented multi-threaded downloading of poster images, resizing to 32x32 pixels and preprocessing using MobileNetV2 pipeline.
    *   Logged failed downloads for inspection.
    *   Extracted embeddings as numerical representations of visual content.

*   **Sentiment Analysis:**
    *   Incorporated sentiment analysis for movie descriptions using TextBlob (NLP library).
    *   Computed sentiment polarity scores (from -1 to 1) to capture the emotional tone of movie overviews.
    *   Included sentiment scores as features in movie embeddings.

*   **Feature Normalization:**
    *   Performed Min-Max Scaling for feature normalization to ensure compatibility between movie and song features.
    *   Normalized movie attributes like `popularity` and `runtime`.
    *   Normalized Spotify features like `popularity` and `tempo`.

*   **Feature Selection:**
    *   Selected relevant features from both datasets based on impact and alignment:
        *   **Movie Features:**
            *   `vote_average`
            *   `vote_count`
            *   `revenue`
            *   `runtime`
            *   `popularity_normalized`
            *   `runtime_normalized`
            *   `overview_sentiment`
            *   `poster_features`
        *   **Spotify Features:**
            *   `popularity_normalized`
            *   `tempo_normalized`
            *   `danceability`
            *   `energy`
            *   `loudness`
            *   `valence`
            *   `speechiness`
            *   `acousticness`
            *   `instrumentalness`

*   **Dataset Merging and Embedding Space:**
    *   Utilized an embedding space approach for merging datasets and finding similarities.
    *   Developed an embedding construction pipeline to combine selected features into a unified representation.
    *   Concatenated movie and Spotify feature embeddings to create training records.

*   **Similarity Calculation:**
    *   Employed cosine similarity to measure alignment between song and movie embeddings.
    *   Compared each song embedding against all movie embeddings to compute similarity scores.
    *   Scores quantify the alignment between songs and movies for ranking recommendations.

*   **Recommendation Generation:**
    *   Selected the top 5 movies with the highest cosine similarity scores for each song as recommendations.
    *   Appended similarity scores to the movie dataset and extracted details of top-ranked movies.
    *   Retained metadata from input Spotify songs (track\_name, tempo, popularity) for output and UI functionality (filtering).

## Predictions

The prediction phase generates movie recommendations for a given song input through these steps:

1.  **Input Processing:**
    *   Process input song features (metadata, audio embeddings) for compatibility with the model.
    *   Incorporate metadata (track\_name, tempo, popularity, etc.) into the pipeline.

2.  **Similarity Calculation:**
    *   Compare song embedding against all movie embeddings using cosine similarity.
    *   Calculate similarity scores to measure alignment between the input song and each movie.

3.  **Top-5 Recommendations:**
    *   Select the top 5 movies with the highest similarity scores.
    *   Rank recommended movies based on similarity scores (most relevant first).

4.  **Result Formatting:**
    *   Output includes movie details (title, overview, popularity, genres).
    *   Includes input song metadata (track\_name, tempo, popularity) for context.
    *   Offers additional filters/attributes (retained from preprocessing) for user customization.

Future improvements include integrating metrics like precision and recall and user feedback mechanisms to evaluate and enhance recommendation relevance, potentially combining cosine similarity with machine learning models.

## Results

The system successfully generates movie recommendations based on embedding similarity between Spotify track features and movie attributes.

**Example Recommendations:**

*   **Song:** "Hold On" by Chord Overstreet
    ```
    Top 5 Recommended Movies:
    - The Lord of the Rings: The Fellowship of the Ring (0.796)
    - The Dark Knight Rises (0.796)
    - The Shawshank Redemption (0.779)
    - The Lord of the Rings: The Two Towers (0.771)
    - Black Panther (0.750)
    ```

*   **Song:** "Days I Will Remember" by Tyrone Wells
    ```
    Top 5 Recommended Movies:
    - The Dark Knight Rises (0.834)
    - Black Panther (0.791)
    - The Wolf of Wall Street (0.783)
    - The Lord of the Rings: The Fellowship of the Ring (0.776)
    - The Lord of the Rings: The Two Towers (0.748)
    ```

**User Interface (UI):**

The UI is designed for intuitive user experience:

*   Users select a song from a dropdown menu.
*   The system displays top recommended movies with similarity scores.
*   A toggle option allows users to view detailed recommendations.
*   Provides an interactive and insightful tool (see Fig. 4 in the paper for UI example).

Overall, the results demonstrate the effectiveness of the approach. Similarity scores align well with the mood and vibe of input songs, and recommended movies are thematically and stylistically compatible. The integration of poster embeddings and sentiment analysis effectively captures visual and emotional movie essence, while audio features ensure connection with songs. This shows the potential of cross-domain embeddings for context-aware, personalized entertainment curation.

## References

*   [1] Maharshipandya. (2023). Spotify Tracks Dataset. Hugging Face. Retrieved from [https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
*   [2] Asaniczka. (2023). TMDb Movies Dataset 2023 (930K+ Movies). Kaggle. Retrieved from [https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
*   [3] Stratoflow. (2022). How to Build a Recommendation System: Explained Step by Step. Retrieved from [https://stratoflow.com/how-to-build-recommendation-system/](https://stratoflow.com/how-to-build-recommendation-system/)
*   [4] GeeksforGeeks. (2023). What are Recommender Systems? Retrieved from [https://www.geeksforgeeks.org/what-are-recommender-systems/](https://www.geeksforgeeks.org/what-are-recommender-systems/)
*   [5] Chen, C.-M., Wang, C.-J., Tsai, M.-F., & Yang, Y.-H. (2019). Collaborative Similarity Embedding for Recommender Systems. Retrieved from [https://arxiv.org/pdf/1902.06188](https://arxiv.org/pdf/1902.06188)
```
