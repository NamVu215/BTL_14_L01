import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Bước 1: Đọc và làm sạch dữ liệu
# Đọc dữ liệu từ các file CSV đã cung cấp
links = pd.read_csv('C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/links.csv')
movies = pd.read_csv('C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/movies.csv')
ratings = pd.read_csv('C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/ratings.csv')
tags = pd.read_csv('C:/Users/TonyZ/Desktop/BTL ML/ML14/ml-latest-small/tags.csv')

# Kiểm tra và làm sạch dữ liệu
# Loại bỏ các hàng có giá trị thiếu trong tập links
links_cleaned = links.dropna()

# Điền giá trị rỗng cho các thể loại bị thiếu trong tập movies (nếu có)
movies['genres'] = movies['genres'].fillna('')

# Chuyển cột 'genres' thành danh sách các thể loại cho mỗi phim
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

# Tạo một cột chứa các thể loại phim dưới dạng chuỗi để dễ dàng tính toán
movies['genres_str'] = movies['genres'].apply(lambda x: ' '.join(x))

# Kiểm tra dữ liệu đã làm sạch
# print("Dữ liệu movies sau khi làm sạch:")
# print(movies.head())

# Bước 2: Xây dựng mô hình gợi ý dựa trên nội dung (Content-Based Filtering)

# Khởi tạo CountVectorizer để chuyển đổi thể loại phim thành dạng ma trận đếm
count_vectorizer = CountVectorizer()
genre_matrix = count_vectorizer.fit_transform(movies['genres_str'])

# Tính toán độ tương đồng cosine giữa các phim
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Tạo hàm để lấy danh sách các phim tương tự dựa trên độ tương đồng
def get_recommendations(title, cosine_sim=cosine_sim):
    # Lấy chỉ số của phim dựa vào tiêu đề
    idx = movies[movies['title'] == title].index[0]

    # Lấy danh sách các phim và độ tương đồng tương ứng
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sắp xếp phim theo độ tương đồng, từ cao đến thấp
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Lấy top 10 phim tương tự (bỏ qua phim hiện tại)
    sim_scores = sim_scores[1:11]

    # Lấy chỉ số phim từ danh sách
    movie_indices = [i[0] for i in sim_scores]

    # Trả về danh sách các phim tương tự
    return movies['title'].iloc[movie_indices]

# Thử nghiệm với một phim cụ thể
print("\nCác phim gợi ý tương tự với 'Toy Story (1995)':")
print(get_recommendations('Toy Story (1995)'))
