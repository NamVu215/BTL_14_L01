from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

def content_based_recommendation(movie_title, movies, n=10):
    """Gợi ý dựa trên nội dung."""
    if movie_title not in movies['title'].values:
        raise ValueError(f"Phim '{movie_title}' không tồn tại trong dữ liệu.")
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

def collaborative_recommendation(user_id, ratings, movies, n=10):
    """Gợi ý dựa trên cộng tác."""
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    
    all_movie_ids = ratings['movieId'].unique()
    rated_movies = ratings[ratings['userId'] == user_id]['movieId']
    unrated_movies = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movies.values]
    
    recommendations = []
    for movie_id in unrated_movies:
        pred = model.predict(user_id, movie_id)
        recommendations.append((movie_id, pred.est))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [rec[0] for rec in recommendations[:n]]
    return movies[movies['movieId'].isin(top_movie_ids)]

def hybrid_recommendation(user_id, movie_title, movies, ratings, weight_cb=0.7, weight_cf=0.3, n=10):
    """Gợi ý kết hợp giữa Content-Based và Collaborative Filtering."""
    cb_recs = content_based_recommendation(movie_title, movies, n=n)
    cf_recs = collaborative_recommendation(user_id, ratings, movies, n=n)
    
    cb_recs['cb_score'] = 1.0
    cf_recs['cf_score'] = cf_recs['movieId'].apply(lambda x: SVD().predict(user_id, x).est)
    
    hybrid_recs = pd.merge(cb_recs, cf_recs, on='movieId', how='outer')
    hybrid_recs = pd.merge(hybrid_recs, movies[['movieId', 'title']], on='movieId', how='left')
    hybrid_recs['final_score'] = (
        weight_cb * hybrid_recs['cb_score'].fillna(0) +
        weight_cf * hybrid_recs['cf_score'].fillna(0)
    )
    return hybrid_recs.sort_values('final_score', ascending=False).head(n)
