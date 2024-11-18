import os
import requests
import pandas as pd

API_KEY = '92c9e60ee91387c5bdb0e27d79854623'  # Thay bằng API Key của bạn

def fetch_tmdb_details(tmdb_id):
    """Gọi API của TMDb để lấy thông tin phim."""
    url = f'https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}&language=en-US'
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'overview': data.get('overview', ''),
                'genres': ' '.join([genre['name'] for genre in data.get('genres', [])]),
                'tagline': data.get('tagline', '')
            }
        else:
            print(f"Lỗi khi gọi API cho tmdbId={tmdb_id}: {response.status_code}")
            return {'overview': '', 'genres': '', 'tagline': ''}
    except requests.exceptions.RequestException as e:
        print(f"Timeout hoặc lỗi khi gọi API cho tmdbId={tmdb_id}: {e}")
        return {'overview': '', 'genres': '', 'tagline': ''}

def load_tmdb_data(links_file, cache_file='tmdb_cache.csv', limit=20):
    """Lấy dữ liệu TMDb từ API và lưu cache."""
    links = pd.read_csv(links_file)
    links = links.dropna(subset=['tmdbId'])  # Loại bỏ các dòng có tmdbId là NaN
    links = links[links['tmdbId'].apply(lambda x: str(x).isdigit())]  # Giữ lại các dòng có tmdbId là số
    links = links.head(limit)  # Giới hạn số dòng để thử nghiệm

    if os.path.exists(cache_file):
        print("Đang tải dữ liệu từ cache...")
        tmdb_df = pd.read_csv(cache_file)
    else:
        print("Đang gọi API TMDb...")
        tmdb_details = []
        for _, row in links.iterrows():
            tmdb_id = int(row['tmdbId'])  # Chuyển tmdbId thành số nguyên
            details = fetch_tmdb_details(tmdb_id)
            tmdb_details.append(details)
        tmdb_df = pd.DataFrame(tmdb_details)
        tmdb_df.to_csv(cache_file, index=False)
    return tmdb_df
