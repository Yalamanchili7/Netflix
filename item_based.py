import pandas as pd
import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def create_item_based_recommender(user_item_matrix):
    """Create item-based collaborative filtering model"""
    try:
        # Convert the user-item matrix to a sparse matrix
        sparse_user_item = csr_matrix(user_item_matrix.values)
        
        # Train the model
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
        model_knn.fit(sparse_user_item.T)  # Transpose for item-based
        
        return model_knn
    except Exception as e:
        st.error(f"Error in create_item_based_recommender: {str(e)}")
        return None

def get_similar_movies(movie_id, user_item_matrix, model_knn, movies, n=10):
    """Get similar movies based on item-based collaborative filtering"""
    try:
        # Check if movie_id exists in the matrix
        if movie_id not in user_item_matrix.columns:
            st.error(f"Movie ID {movie_id} is not in the user-item matrix. It may not have enough ratings.")
            return pd.DataFrame()
        
        # Get the index of the movie in the matrix
        movie_idx = user_item_matrix.columns.get_loc(movie_id)
        
        # Get movie vector
        sparse_ui = csr_matrix(user_item_matrix.values)
        movie_vector = sparse_ui.T[movie_idx].reshape(1, -1)
        
        # Get the k nearest neighbors
        try:
            distances, indices = model_knn.kneighbors(
                movie_vector, 
                n_neighbors=min(n+1, len(user_item_matrix.columns))
            )
            
            # Convert indices to movie IDs (skip the first one as it's the input movie)
            similar_movie_indices = indices.flatten()[1:]
            
            # Handle empty results
            if len(similar_movie_indices) == 0:
                st.warning("No similar movies found.")
                return pd.DataFrame()
                
            similar_movie_ids = [user_item_matrix.columns[idx] for idx in similar_movie_indices]
            similarity_scores = [1 - dist for dist in distances.flatten()[1:]]
            
            # Get movie details
            similar_movies = movies[movies['movieId'].isin(similar_movie_ids)].copy()
            
            # Add similarity score
            similarity_dict = dict(zip(similar_movie_ids, similarity_scores))
            similar_movies['similarity'] = similar_movies['movieId'].map(similarity_dict)
            
            # Sort by similarity
            similar_movies = similar_movies.sort_values('similarity', ascending=False)
            
            return similar_movies
            
        except Exception as e:
            st.error(f"Error in kNN calculation: {str(e)}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error in get_similar_movies: {str(e)}")
        return pd.DataFrame()
