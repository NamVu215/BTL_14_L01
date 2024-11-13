import pandas as pd

# Đọc dữ liệu từ các file CSV đã cung cấp
links = pd.read_csv('C:/Users/Administrator/Desktop/BTL ML/ML14/ml-latest-small/links.csv')
movies = pd.read_csv('C:/Users/Administrator/Desktop/BTL ML/ML14/ml-latest-small/movies.csv')
ratings = pd.read_csv('C:/Users/Administrator/Desktop/BTL ML/ML14/ml-latest-small/ratings.csv')
tags = pd.read_csv('C:/Users/Administrator/Desktop/BTL ML/ML14/ml-latest-small/tags.csv')

# Kiểm tra xem có giá trị thiếu trong từng tập dữ liệu không và làm sạch nếu cần

# Kiểm tra giá trị thiếu trong tập dữ liệu links
links_missing = links.isnull().sum()

# Loại bỏ các hàng có giá trị thiếu trong tập links
links_cleaned = links.dropna()

# Kiểm tra giá trị thiếu trong tập dữ liệu movies
movies_missing = movies.isnull().sum()

# Điền giá trị rỗng cho các thể loại bị thiếu trong tập movies (nếu có)
movies['genres'] = movies['genres'].fillna('')

# Chuyển cột 'genres' thành danh sách các thể loại cho mỗi phim
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

# Kiểm tra giá trị thiếu trong tập dữ liệu ratings
ratings_missing = ratings.isnull().sum()

# Loại bỏ các hàng có giá trị thiếu trong tập ratings
ratings_cleaned = ratings.dropna()

# Kiểm tra giá trị thiếu trong tập dữ liệu tags
tags_missing = tags.isnull().sum()

# Loại bỏ các hàng có giá trị thiếu trong tập tags
tags_cleaned = tags.dropna()

# Hiển thị dữ liệu đã được làm sạch và thông tin về các giá trị thiếu
links_cleaned_head = links_cleaned.head()    # Hiển thị vài dòng đầu tiên của tập links đã làm sạch
movies_cleaned_head = movies.head()          # Hiển thị vài dòng đầu tiên của tập movies đã làm sạch
ratings_cleaned_head = ratings.head()        # Hiển thị vài dòng đầu tiên của tập ratings đã làm sạch
tags_cleaned_head = tags.head()              # Hiển thị vài dòng đầu tiên của tập tags đã làm sạch

# Kết quả trả về bao gồm thông tin về các giá trị thiếu và dữ liệu đã được làm sạch
(links_missing, movies_missing, ratings_missing, tags_missing,
 links_cleaned_head, movies_cleaned_head, ratings_cleaned_head, tags_cleaned_head)

# In ra kết quả để hiển thị trên terminal
print("Giá trị thiếu trong tập dữ liệu links:", links_missing)
print("Giá trị thiếu trong tập dữ liệu movies:", movies_missing)
print("Giá trị thiếu trong tập dữ liệu ratings:", ratings_missing)
print("Giá trị thiếu trong tập dữ liệu tags:", tags_missing)

print("\nDữ liệu đã làm sạch - links:")
print(links_cleaned_head)

print("\nDữ liệu đã làm sạch - movies:")
print(movies_cleaned_head)

print("\nDữ liệu đã làm sạch - ratings:")
print(ratings_cleaned_head)

print("\nDữ liệu đã làm sạch - tags:")
print(tags_cleaned_head)
