import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from surprise import SVD, Dataset, Reader

# Bước 1: Đọc dữ liệu
links = pd.read_csv('C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/links.csv')
movies = pd.read_csv('C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/movies.csv')
ratings = pd.read_csv('C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/ratings.csv')
tags = pd.read_csv('C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/tags.csv')

# Bước 2: Tích hợp dữ liệu từ `links.csv` và `tags.csv`
# Kết hợp tags vào bảng movies
tags['tag'] = tags['tag'].fillna('')

# Gộp tất cả các tag vào một chuỗi ký tự cho mỗi movieId
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# Kết hợp tag và thông tin từ links vào movies
movies = pd.merge(movies, movie_tags, on='movieId', how='left')
movies = pd.merge(movies, links, on='movieId', how='left')

# Thêm cột 'content' kết hợp thể loại và tag
movies['tags'] = movies['tag'].fillna('')
movies['genres'] = movies['genres'].fillna('')
movies['content'] = movies['genres'] + ' ' + movies['tags']

# Bước 3: Tính toán độ tương đồng Content-Based Filtering
count_vectorizer = CountVectorizer()
content_matrix = count_vectorizer.fit_transform(movies['content'])
cosine_sim = cosine_similarity(content_matrix, content_matrix)

# Hàm gợi ý dựa trên nội dung
def content_based_recommendation(movie_title, n=10):
    if movie_title not in movies['title'].values:
        raise ValueError(f"Phim '{movie_title}' không tồn tại trong dữ liệu.")
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

# Bước 4: Chuẩn bị Collaborative Filtering
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
cf_model = SVD()
cf_model.fit(trainset)

# Hàm gợi ý dựa trên cộng tác
def collaborative_recommendation(user_id, n=10):
    all_movie_ids = ratings['movieId'].unique()
    rated_movies = ratings[ratings['userId'] == user_id]['movieId']
    unrated_movies = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movies.values]
    recommendations = []
    for movie_id in unrated_movies:
        pred = cf_model.predict(user_id, movie_id)
        recommendations.append((movie_id, pred.est))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [rec[0] for rec in recommendations[:n]]
    return movies[movies['movieId'].isin(top_movie_ids)]

# Bước 5: Hybrid Recommendation
def hybrid_recommendation(user_id, movie_title, n=10, weight_cb=0.5, weight_cf=0.5):
    # Gợi ý dựa trên nội dung
    cb_recs = content_based_recommendation(movie_title, n=n)
    cb_recs = cb_recs.copy()
    cb_recs['cb_score'] = 1.0  # Điểm mặc định cho gợi ý dựa trên nội dung

    # Gợi ý dựa trên cộng tác
    cf_recs = collaborative_recommendation(user_id, n=n)
    cf_recs = cf_recs.copy()
    cf_recs['cf_score'] = cf_recs['movieId'].apply(lambda x: cf_model.predict(user_id, x).est)

    # Gộp hai kết quả
    hybrid_recs = pd.merge(cb_recs, cf_recs, on='movieId', how='outer')
    hybrid_recs = pd.merge(hybrid_recs, movies[['movieId', 'title']], on='movieId', how='left')
    hybrid_recs['final_score'] = (
        weight_cb * hybrid_recs['cb_score'].fillna(0) +
        weight_cf * hybrid_recs['cf_score'].fillna(0)
    )
    hybrid_recs = hybrid_recs.sort_values('final_score', ascending=False)
    return hybrid_recs[['movieId', 'title', 'final_score']].head(n)

# Bước 6: Thử nghiệm
user_id = 1  # Thay ID người dùng để thử nghiệm
movie_title = 'Toy Story (1995)'  # Thay phim để thử nghiệm

print(f"\nGợi ý kết hợp cho người dùng {user_id} với phim '{movie_title}':")
hybrid_results = hybrid_recommendation(user_id=user_id, movie_title=movie_title, n=10)
print(hybrid_results)

# Lấy thông tin từ movies gốc dựa trên các chỉ số
original_rows = movies.iloc[[0, 6, 14, 1, 7, 19, 5, 4, 3, 2]]
print(original_rows)
