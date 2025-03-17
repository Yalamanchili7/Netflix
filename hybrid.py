import pandas as pd
import streamlit as st

def get_hybrid_recommendations(user_id, movies, predicted_df, user_item_matrix, movie_id=None, n=10):
    """Get hybrid recommendations combining collaborative filtering and content-based"""
    try:
        # Check if user exists
        if user_id not in user_item_matrix.index:
            return pd.DataFrame()
        
        # Get collaborative filtering recommendations
        from recommendation.collaborative import get_cf_recommendations
        cf_recs = get_cf_recommendations(user_id, predicted_df, user_item_matrix, movies, n=n*2)
        
        # If no movie_id provided, return CF recommendations
        if movie_id is None:
            return cf_recs.head(n)
        
        # Get the user's ratings
        user_ratings = user_item_matrix.loc[user_id]
        
        # If the user has rated the movie, use this rating as weight
        if movie_id in user_ratings.index and user_ratings[movie_id] > 0:
            movie_weight = user_ratings[movie_id] / 5.0  # Normalize to 0-1
        else:
            movie_weight = 0.5  # Default weight
        
        # Blend the scores
        # For simplicity, we'll just boost movies with similar genres
        input_movie = movies[movies['movieId'] == movie_id]
        
        if not input_movie.empty:
            input_genres = set(input_movie.iloc[0]['genres'].split('|'))
            
            # Calculate genre similarity for each recommendation
            cf_recs['genre_match'] = cf_recs['genres'].apply(
                lambda x: len(set(x.split('|')) & input_genres) / len(input_genres) 
                if len(input_genres) > 0 else 0
            )
            
            # Adjust predicted rating based on genre similarity
            cf_recs['hybrid_score'] = (
                (1 - movie_weight) * cf_recs['predicted_rating'] + 
                movie_weight * 5 * cf_recs['genre_match']
            )
            
            # Sort by hybrid score
            hybrid_recs = cf_recs.sort_values('hybrid_score', ascending=False).head(n)
            return hybrid_recs
        else:
            return cf_recs.head(n)
    
    except Exception as e:
        st.error(f"Error in get_hybrid_recommendations: {str(e)}")
        return pd.DataFrame()
