
import streamlit as st
import pandas as pd

# Load the recommendations DataFrame
recommendations_df = pd.read_csv("all_recommendations.csv")

# Streamlit app logic
st.title("Music2Movie Recommendation System")

# Song selection dropdown
selected_song = st.selectbox("Select a Song:", recommendations_df['song_track_name'].unique())

# Filter recommendations for the selected song
filtered_df = recommendations_df[recommendations_df['song_track_name'] == selected_song]

# Display recommendations for the selected song
st.write(f"Top recommendations for the song '{selected_song}':")
st.dataframe(filtered_df[['title', 'similarity_score']])  # Display movie title and similarity score

# Optional: Include additional song and movie details
if st.checkbox("Show detailed recommendations"):
    st.write("Detailed recommendations:")
    st.dataframe(filtered_df)
