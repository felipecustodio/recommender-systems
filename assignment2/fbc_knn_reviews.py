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
FBC-kNN
Using movie reviews
"""

algorithm = "FBC-kNN-reviews"


def FBC_kNN_reviews():
    pass


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
        prediction = FBC_kNN_reviews(ratings, user-1, movie-1)
        error_check(prediction, id)

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
