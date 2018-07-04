#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0284 - Recommender Systems - 2018/1
Assignment 2
ICMC - University of Sao Paulo
Professor Marcelo Manzato
Students:
Felipe Scrochio Custódio - 9442688
Lucas Antognoni de Castro - 8936951
"""

import pandas
import csv
import math
import numpy as np
import progressbar
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from termcolor import colored

"""
Probabilistic Method
Using movie genres
"""

algorithm = "probabilistic-genres"


def naiveBayes(user, movie, train_data, movies_data):

    attributes = data.loc[data['user_id'] == user, 'movie_id':'rating']
    genres = movies_data.loc[movies_data['movie_id'] == movie, 'movie_id':'genres']
    genres = genres.split('|')

    p_of_g = {}
    # calcular a probabilidade de ocorrência de cada gênero
    total_count = 0
    for genre in genres:
        p_of_g[genre] = 0
        for row in movies_data.itertuples():
            for inner_genre in (row['genres']).split('|'):
                if (inner_genre == genre):
                    p_of_g[genre] += 1
                total_count += 1

    for genre in genres:
        p_of_g[genre] = p_of_g[genre] / total_count

    # Hypothesis probability
    p_of_v = np.zeros((5))

    for i in range(5):
        if i + 1 in movie_ratings_frequence:
            p_of_v[i] = (movie_ratings_frequence[i + 1] / (sum(movie_ratings_frequence.values())))
        else:
            p_of_v[i] = 0

    # Conditional probability P(ai|vj)
    p_a_v = np.zeros((len(genres), 5))

    index = 0
    for genre in genres:
        for rating in range(5):

            a = len(data.loc[(data['movie_id'] == row.movie_id) & (data['rating'] == col + 1)])
            b = len(data.loc[(data['movie_id'] == movie) & (data['rating'] == col + 1)])

            if a == 0 or b == 0:
                p_a_v[index][col] = 0
            else:
                p_a_v[index][col] = b / a

        index += 1

    p = np.zeros((5))
    for i in range(5):
        p[i] = p_of_v[i] * np.prod(p_a_v[:,i])

    return (np.argmax(p) + 1)


# read dataset
movies_data = pandas.read_csv("csv/movies_data.csv")
users_data = pandas.read_csv("csv/users_data.csv")
train_data = pandas.read_csv("csv/train_data.csv")
test_data = pandas.read_csv("csv/test_data.csv")
reviews_data = pandas.read_csv("csv/movie_reviews.csv")

# write results CSV header
results_csv = open('results/' + algorithm + '.csv', 'w', newline='')
results_writer = csv.writer(results_csv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
results_writer.writerow(['id', 'rating'])

# run algorithm for test cases
counter = 0
times = []
with progressbar.ProgressBar(max_value=3970) as bar:
    for row in test_data.itertuples():
        start = timer()

        # get parameters
        id = getattr(row, "id")
        user = getattr(row, "user_id")
        movie = getattr(row, "movie_id")

        # run prediction algorithm
        prediction = naiveBayes(row.id, row.user_id, row.movie_id, train_data)

        # write new row in csv file
        end = timer()
        time_elapsed = end - start
        times.append(time_elapsed)
        results_writer.writerow([id, prediction])

        counter += 1
        bar.update(counter)


# plot time elapsed
plt.scatter(range(len(times)), times, s=1, c="#F67280")
plt.xlabel("Test case (ID)")
plt.ylabel("Time elapsed (seconds)")
plt.title(algorithm)
# get current figure
figure = plt.gcf()
# # 800 x 600
figure.set_size_inches(8, 6)
# # save with high DPI
plt.savefig("plots/time_" + algorithm + ".png", dpi=100)
plt.clf()
