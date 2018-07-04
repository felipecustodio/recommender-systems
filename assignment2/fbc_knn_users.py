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
Using movie reviews
"""

algorithm = "FBC-kNN-reviews"

def error_check(prediction, id):
    if (prediction < 1 or prediction > 5 or math.isnan(prediction)):
        text = colored('ERROR: ', 'red', attrs=['reverse', 'blink'])
        print(text + "prediction {} at position {}".format(prediction, id))


def jaccard_index(first_user, second_user):

    number_of_attributes_first = len(first_user)
    number_of_attributes_second = len(second_user)

    intersection = len(list(set(first_user) & set(second_user)))

    return intersection / (number_of_attributes_first + number_of_attributes_second - intersection)


def FBC_kNN_users(q_id, user, item, neighbors, similarity, data):
    
    user = data.loc[data['user_id'] == user]
    similar_itens = np.copy(similarity[item - 1, :])
    sorted_indexes = np.argsort(similar_itens)[-1:-neighbors:-1]

    sum_a = 0
    sum_b = 0

    for index in sorted_indexes:

        rated_movie = user.loc[user['movie_id'] == (index + 1)]

        if rated_movie.empty == False:
            sum_a += similar_itens[index] * rated_movie.rating.values[0]
            sum_b += similar_itens[index]
        
    if (sum_a == 0) or (sum_b == 0):
        print("%d, 0"  % (q_id))
    else:
        print("%d,%f"  % (q_id,sum_a / sum_b))


# read dataset
users_data = pandas.read_csv("csv/users_data.csv")
train_data = pandas.read_csv("csv/train_data.csv")
test_data = pandas.read_csv("csv/test_data.csv")

# write results CSV header
# results_csv = open('results/' + algorithm + '.csv', 'w', newline='')
# results_writer = csv.writer(results_csv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
# results_writer.writerow(['id', 'rating'])

# similarity_matrix = np.zeros((users_data.shape[0], users_data.shape[0]))

users = np.sort(users_data.user_id.unique())
similarity_matrix = np.zeros((users[-1], users[-1]))

for row in users_data.itertuples():

    user_index = getattr(row, "Index")
    user_attributes = []
    user_attributes.append(getattr(row, "gender"))
    user_attributes.append(getattr(row, "age"))
    user_attributes.append(getattr(row, "occupation"))
    user_attributes.append(getattr(row, "zip_code"))
    
    new_users = users_data

    for new_row in new_users.itertuples():
        
        new_user_index = getattr(new_row, "Index")
        new_user_attributes = []
        new_user_attributes.append(getattr(new_row, "gender"))
        new_user_attributes.append(getattr(new_row, "age"))
        new_user_attributes.append(getattr(new_row, "occupation"))
        new_user_attributes.append(getattr(new_row, "zip_code"))

        if user_index == new_user_index:
            similarity_matrix[user_index][new_user_index] = 0.0
        else:
            similarity_matrix[user_index][new_user_index] = jaccard_index(user_attributes, new_user_attributes)



# # run algorithm for test cases
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
#         prediction = FBC_kNN_users(ratings, user-1, movie-1)
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
    prediction = FBC_kNN_users(id, user, movie, 50, similarity_matrix, train_data)
    