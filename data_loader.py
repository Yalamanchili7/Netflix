import pandas as pd
import numpy as np
import streamlit as st

def load_data(ratings_path, movies_path, sample_size=None):
    """Load ratings and movies data from MovieLens dataset"""
    try:
        # Load ratings
        ratings = pd.read_csv(ratings_path)
        st.write(f"Successfully loaded ratings: {ratings.shape}")
        
        # Sample data if requested
        if sample_size and len(ratings) > sample_size:
            ratings = ratings.sample(sample_size, random_state=42)
            st.write(f"Sampled ratings: {ratings.shape}")
        
        # Load movies
        movies = pd.read_csv(movies_path)
        st.write(f"Successfully loaded movies: {movies.shape}")
        
        return ratings, movies
    
    except Exception as e:
        st.error(f"Error in load_data: {str(e)}")
        return None, None

def preprocess_data(ratings, movies, min_user_ratings=10, min_movie_ratings=10):
    """Preprocess data for recommendation algorithms"""
    try:
        # Merge ratings with movie data
        data = pd.merge(ratings, movies, on='movieId')
        
        # Filter out users with few ratings
        user_counts = data.groupby('userId')['rating'].count()
        active_users = user_counts[user_counts >= min_user_ratings].index
        data = data[data['userId'].isin(active_users)]
        
        # Filter out movies with few ratings
        movie_counts = data.groupby('movieId')['rating'].count()
        active_movies = movie_counts[movie_counts >= min_movie_ratings].index
        data = data[data['movieId'].isin(active_movies)]
        
        st.write(f"After preprocessing: {len(active_users)} users, {len(active_movies)} movies")
        
        # Create user-item matrix
        user_item_matrix = data.pivot_table(index='userId', columns='movieId', values='rating')
        
        # Fill missing values with 0
        user_item_matrix_filled = user_item_matrix.fillna(0)
        
        return data, user_item_matrix, user_item_matrix_filled
    
    except Exception as e:
        st.error(f"Error in preprocess_data: {str(e)}")
        return None, None, None
