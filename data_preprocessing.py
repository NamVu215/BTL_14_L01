import pandas as pd

def preprocess_movies(movies_file, tmdb_df, tags_file=None):
    """Tiền xử lý dữ liệu từ movies và tags."""
    movies = pd.read_csv(movies_file)
    movies = pd.concat([movies, tmdb_df], axis=1).drop_duplicates(subset=['movieId']).reset_index(drop=True)
    
    if tags_file:
        tags = pd.read_csv(tags_file)
        tags['tag'] = tags['tag'].fillna('')
        movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        movies = pd.merge(movies, movie_tags, on='movieId', how='left')
    
    # Đảm bảo các cột cần thiết tồn tại
    for col in ['genres', 'overview', 'tagline', 'tags']:
        if col not in movies.columns:
            movies[col] = ''
        else:
            movies[col] = movies[col].fillna('')
    
    # Tạo cột nội dung
    movies['content'] = (
        movies['genres'] + ' ' +
        movies['tags'] + ' ' +
        movies['overview'] + ' ' +
        movies['tagline']
    )
    return movies

def preprocess_ratings(ratings_file, movie_ids):
    """Lọc ratings chỉ giữ các phim trong danh sách movie_ids."""
    ratings = pd.read_csv(ratings_file)
    return ratings[ratings['movieId'].isin(movie_ids)]
