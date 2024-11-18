from tmdb_api import load_tmdb_data
from data_preprocessing import preprocess_movies, preprocess_ratings
from recommenders import content_based_recommendation, collaborative_recommendation, hybrid_recommendation

# Đường dẫn tệp
MOVIES_FILE = 'path/to/movies.csv'
RATINGS_FILE = 'path/to/ratings.csv'
TAGS_FILE = 'path/to/tags.csv'
LINKS_FILE = 'path/to/links.csv'

# Lấy dữ liệu TMDb
tmdb_df = load_tmdb_data(LINKS_FILE)

# Tiền xử lý dữ liệu
movies = preprocess_movies(MOVIES_FILE, tmdb_df, TAGS_FILE)
ratings = preprocess_ratings(RATINGS_FILE, movies['movieId'])

# Gợi ý
user_id = 1
movie_title = 'Toy Story (1995)'

print("\nGợi ý Content-Based:")
print(content_based_recommendation(movie_title, movies))

print("\nGợi ý Collaborative:")
print(collaborative_recommendation(user_id, ratings, movies))

print("\nGợi ý Hybrid:")
print(hybrid_recommendation(user_id, movie_title, movies, ratings))
