from tmdb_api import load_tmdb_data
from data_preprocessing import preprocess_movies, preprocess_ratings
from recommenders import content_based_recommendation, collaborative_recommendation, hybrid_recommendation
from surprise import SVD, Dataset, Reader

# Đường dẫn tệp
MOVIES_FILE = 'C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/movies.csv'
RATINGS_FILE = 'C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/ratings.csv'
TAGS_FILE = 'C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/tags.csv'
LINKS_FILE = 'C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/links.csv'

# Lấy dữ liệu TMDb
print("Bắt đầu tải dữ liệu TMDb...")
tmdb_df = load_tmdb_data(LINKS_FILE, limit=20)

# Tiền xử lý dữ liệu
print("Tiền xử lý dữ liệu...")
movies = preprocess_movies(MOVIES_FILE, tmdb_df, TAGS_FILE)
ratings = preprocess_ratings(RATINGS_FILE, movies['movieId'])

# Huấn luyện model Collaborative Filtering
reader = Reader(rating_scale=(0.5, 5.0))
ratings_data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = ratings_data.build_full_trainset()
cf_model = SVD()
cf_model.fit(trainset)

# Gợi ý
user_id = 1
movie_title = 'Toy Story (1995)'

print("\nGợi ý Content-Based:")
print(content_based_recommendation(movie_title, movies))

print("\nGợi ý Collaborative:")
print(collaborative_recommendation(user_id, ratings, movies, cf_model))

print("\nGợi ý Hybrid:")
print(hybrid_recommendation(user_id, movie_title, movies, ratings, cf_model))
