import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Bước 1: Chuẩn bị dữ liệu
# Đọc dữ liệu ratings từ file CSV
links = pd.read_csv('C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/links.csv')
movies = pd.read_csv('C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/movies.csv')
ratings = pd.read_csv('C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/ratings.csv')
tags = pd.read_csv('C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/tags.csv')

# Khởi tạo đối tượng Reader và Dataset của Surprise từ DataFrame ratings
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
trainset, testset = train_test_split(data, test_size=0.2)

# Bước 2: Khởi tạo mô hình SVD và huấn luyện
# Sử dụng SVD cho Collaborative Filtering
svd = SVD()
svd.fit(trainset)

# Bước 3: Đánh giá mô hình trên tập kiểm tra
predictions = svd.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE của mô hình Collaborative Filtering: {rmse}")

# Bước 4: Tạo hàm gợi ý phim cho người dùng
# Hàm gợi ý phim cho một người dùng dựa trên mô hình SVD
def recommend_movies(user_id, n_recommendations=10):
    # Lấy danh sách tất cả movieId trong tập ratings
    all_movie_ids = ratings['movieId'].unique()
    
    # Lấy danh sách các phim mà người dùng đã đánh giá
    rated_movies = ratings[ratings['userId'] == user_id]['movieId']
    
    # Lọc ra các phim mà người dùng chưa đánh giá
    unrated_movies = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movies.values]
    
    # Dự đoán rating cho các phim chưa đánh giá và sắp xếp theo thứ tự giảm dần
    recommendations = []
    for movie_id in unrated_movies:
        pred = svd.predict(user_id, movie_id)
        recommendations.append((movie_id, pred.est))
    
    # Sắp xếp danh sách dựa trên rating dự đoán và lấy top n_recommendations
    recommendations.sort(key=lambda x: x[1], reverse=True)
    recommended_movie_ids = [rec[0] for rec in recommendations[:n_recommendations]]
    
    # Trả về danh sách các phim được gợi ý (dựa trên title)
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
    return recommended_movies[['movieId', 'title']]

# Thử nghiệm gợi ý phim cho một người dùng cụ thể (ví dụ: userId = 1)
print("\nGợi ý phim cho người dùng userId = 1:")
print(recommend_movies(user_id=1))
