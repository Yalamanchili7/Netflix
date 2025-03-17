import pandas as pd
import numpy as np
import streamlit as st
from sklearn.decomposition import TruncatedSVD

def create_collaborative_filtering(user_item_matrix_filled, n_components=50):
    """Create collaborative filtering recommender using SVD"""
    try:
        # Create SVD model
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Fit the model
        latent_matrix = svd.fit_transform(user_item_matrix_filled)
        
        # Get item latent factors
        item_latent_factors = svd.components_
        
        # Compute predicted ratings
        predicted_ratings = np.dot(latent_matrix, item_latent_factors)
        
        # Create DataFrame with predicted ratings
        predicted_df = pd.DataFrame(
            predicted_ratings, 
            index=user_item_matrix_filled.index, 
            columns=user_item_matrix_filled.columns
        )
        
        return predicted_df
    except Exception as e:
        st.error(f"Error in create_collaborative_filtering: {str(e)}")
        return None

def get_cf_recommendations(user_id, predicted_df, user_item_matrix, movies, n=10):
    """Get recommendations for a user based on collaborative filtering"""
    try:
        # Check if user exists
        if user_id not in predicted_df.index:
            return pd.DataFrame()
        
        # Get the user's predicted ratings
        user_ratings = predicted_df.loc[user_id]
        
        # Get the user's already rated movies
        rated_movies = user_item_matrix.loc[user_id]
        rated_movies = rated_movies[rated_movies > 0].index
        
        # Filter out already rated movies
        recommendations = user_ratings[~user_ratings.index.isin(rated_movies)]
        
        # Get top recommendations
        top_recs = recommendations.sort_values(ascending=False).head(n)
        
        # Get movie details
        rec_details = pd.DataFrame({
            'movieId': top_recs.index,
            'predicted_rating': top_recs.values
        })
        
        rec_details = pd.merge(
            rec_details, 
            movies[['movieId', 'title', 'genres']], 
            on='movieId'
        )
        
        return rec_details
    except Exception as e:
        st.error(f"Error in get_cf_recommendations: {str(e)}")
        return pd.DataFrame()
