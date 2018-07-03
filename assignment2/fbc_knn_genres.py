#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0284 - Recommender Systems - 2018/1
Assignment 2
ICMC - University of Sao Paulo
Professor: Marcelo Manzato
Students: Felipe Scrochio Custódio - 9442688
          Lucas Antognoni de Castro - 8936951
"""

import pandas
import csv
import math
import numpy as np
import progressbar
# from matplotlib import pyplot as plt
from timeit import default_timer as timer
from termcolor import colored

"""
FBC-kNN
Using movie genres
"""

algorithm = "FBC-kNN-genres"


def error_check(prediction, id):
    if (prediction < 1 or prediction > 5 or math.isnan(prediction)):
        text = colored('ERROR: ', 'red', attrs=['reverse', 'blink'])
        print(text + "prediction {} at position {}".format(prediction, id))

def jaccard_index(first_movie, second_movie):

    number_of_genres_first = len(first_movie)
    number_of_genres_second = len(second_movie)

    intersection = len(list(set(first_movie) & set(second_movie)))

    return intersection / (number_of_genres_first + number_of_genres_second - intersection)

def FBC_kNN_genres(q_id, user, item, neighbors, similarity, data):
    
    user = data.loc[data['user_id'] == user]
    similar_itens = similarity[item - 1][:]
    sorted_indexes = np.argsort(similar_itens)[0:neighbors]

    sum_a = 0
    sum_b = 0

    for index in sorted_indexes:
        sum_a += similar_itens[index] * user.loc[user['movie_id'] == (index + 1)].rating
        sum_b += similar_itens[index]

    print("%d,%f"  % (q_id,sum_a / sum_b))


# read dataset
movies_data = pandas.read_csv("csv/movies_data.csv")
train_data = pandas.read_csv("csv/train_data.csv")
test_data = pandas.read_csv("csv/test_data.csv")

# write results CSV header
# results_csv = open('results/' + algorithm + '.csv', 'w', newline='')
# results_writer = csv.writer(results_csv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
# results_writer.writerow(['id', 'rating'])


similarity_matrix = np.zeros((movies_data.shape[0], movies_data.shape[0]))

for row in movies_data.itertuples():

    movie_index = getattr(row, "Index")
    movie_genres = getattr(row, "genres").split('|')

    new_movies = movies_data[movie_index:]

    similarity_matrix[movie_index][movie_index] = 0.0

    for new_row in new_movies.itertuples():
        
        new_movie_index = getattr(new_row, "Index")
        new_movie_genres = getattr(new_row, "genres").split('|')

        similarity_matrix[movie_index][new_movie_index] = jaccard_index(movie_genres, new_movie_genres)


# run algorithm for test cases
# counter = 0
# times = []
# with progressbar.ProgressBar(max_value=3970) as bar:
#     for row in test_data.itertuples():
#         start = timer()

#         # get parameters
#         id = getattr(row, "id")
#         user = getattr(row, "user_id")
#         movie = getattr(row, "movie_id")

#         # run prediction algorithm
#         prediction = FBC_kNN_genres(user-1, movie-1)
#         error_check(prediction, id)

#         # write new row in csv file
#         end = timer()
#         time_elapsed = end - start
#         times.append(time_elapsed)
#         results_writer.writerow([id, prediction])

#         counter += 1
#         bar.update(counter)


# # plot time elapsed
# plt.scatter(range(len(times)), times, s=1, c="#F67280")
# plt.xlabel("Test case (ID)")
# plt.ylabel("Time elapsed (seconds)")
# plt.title(algorithm)
# # get current figure
# figure = plt.gcf()
# # # 800 x 600
# figure.set_size_inches(8, 6)
# # # save with high DPI
# plt.savefig("plots/time_" + algorithm + ".png", dpi=100)
# plt.clf()

for row in test_data.itertuples():

    # get parameters
    id = getattr(row, "id")
    user = getattr(row, "user_id")
    movie = getattr(row, "movie_id")

    # run prediction algorithm
    prediction = FBC_kNN_genres(id, user, movie, 20, similarity_matrix, train_data)
    # error_check(prediction, id)


