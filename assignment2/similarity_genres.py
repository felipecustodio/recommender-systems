from __future__ import division
import pandas
import numpy as np

def jaccard_index(first_movie, second_movie):

    number_of_genres_first = len(first_movie)
    number_of_genres_second = len(second_movie)

    intersection = len(list(set(first_movie) & set(second_movie)))

    # print(number_of_genres_first)
    # print(number_of_genres_second)
    # print(intersection)

    return intersection / (number_of_genres_first + number_of_genres_second - intersection)


# read dataset
movies_data = pandas.read_csv("csv/movies_data.csv")    
     
similarity_matrix = np.zeros((movies_data.shape[0], movies_data.shape[0]))

for row in movies_data.itertuples():

    movie_index = getattr(row, "Index")
    movie_genres = getattr(row, "genres").split('|')

    new_movies = movies_data[movie_index:]

    # print(movie_genres)

    similarity_matrix[movie_index][movie_index] = 0.0

    for new_row in new_movies.itertuples():
        
        new_movie_index = getattr(new_row, "Index")
        new_movie_genres = getattr(new_row, "genres").split('|')

        # print(new_movie_genres)

        print(movie_index, new_movie_index)

        similarity_matrix[movie_index][new_movie_index] = jaccard_index(movie_genres, new_movie_genres)

df = pandas.DataFrame(similarity_matrix)
df.to_csv("./csv/similarity_genres.csv")


