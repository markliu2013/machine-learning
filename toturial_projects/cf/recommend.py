"""
通过如下程序提取数据：
@Desc：读取用户的电影数据和评分数据
"""
import pandas as pd

data_path = 'data/ml-latest-small/'
dist_path = 'dist/'

movies = pd.read_csv(data_path + "movies.csv")
ratings = pd.read_csv(data_path + "ratings.csv")
data = pd.merge(movies, ratings, how='left', on='movieId')
data[['userId', 'rating', 'movieId', 'title']].sort_values('userId').to_csv(dist_path + 'data.csv', index=False)
