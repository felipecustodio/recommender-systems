#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0284 - Recommender Systems - 2018/1
Assignment 1
ICMC - University of Sao Paulo
Professor Marcelo Manzato
Student: Felipe Scrochio Custódio - 9442688
"""

import os
import psutil
import operator

import time
import progressbar
from timeit import default_timer as timer

import math
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Algorithms
def user_median_ratings(ratings, user):
    # get only existing ratings
    # filter None eliminates false values
    existing_ratings = list(filter(None, ratings[user]))
    median = np.sum(existing_ratings) / len(existing_ratings)
    return median


def similarity_user(u1, u2):
    pass


def similarity_item(ratings, i, j):
    numerator, denominator, denominator1, denominator2 = (0,0,0,0)
    # find all users that rated i and j (U)
    U = []
    # slice i and j columns
    i_ratings = ratings[:,i]
    j_ratings = ratings[:,j]
    for user_id, rating in enumerate(i_ratings):
        # if both items were rated by user, append to U
        if ((i_ratings[user_id-1] != None) and (j_ratings[user_id-1]) != None):
            U.append(user_id)

    # if users were found, calculate similarity
    if U:
        for user in U:
            # get user ratings for i and j
            rui = i_ratings[user-1]
            ruj = j_ratings[user-1]
            # get user globan median
            ru = user_median_ratings(ratings, user)
            # calculate similarity
            numerator += ((rui - ru) * (ruj - ru))
            denominator1 += (np.power((rui - ru), 2))
            denominator2 += (np.power((ruj - ru), 2))
        denominator = np.sqrt(denominator1) * np.sqrt(denominator2)
        try:
            similarity = float(numerator) / float(denominator)
        except ZeroDivisionError:
            # division by zero probably means there was only
            # one user that rated both items i and j, rating
            # with his median, which yields division by zero
            similarity = -1
    else:
        # no users found, set similarity to least possible
        similarity = -1
    return similarity


def k_most_similar_items(ratings, u, i, k):
    # similarity list has size 'number of items'
    similarities = np.empty((ratings.shape[1]))
    # items that user 'u' rated
    items = ratings[u]
    # find all similarities with item 'i'
    # that 'u' has rated
    for movie_id, item in enumerate(items):
        if (ratings[u-1][movie_id-1] != None):
            similarities[movie_id-1] = similarity_item(ratings, i, movie_id)
    # sort similarities list and get 'k' most similar
    k_biggest = similarities[np.argsort(similarities)[-k:]]
    # filter NaN
    # k_biggest = k_biggest[~np.isnan(k_biggest)]
    # filter values with a treshold of 0.01
    k_biggest = k_biggest[k_biggest > 0.01]
    # we need to know which movie_id has that similarity
    # turn list into dictionary
    k_most_similar = {}
    for similarity in k_biggest:
        # get movie_id that has similarity 'similarity'
        movie_id = (list(similarities).index(similarity)) + 1
        # index similarity by movie_id
        k_most_similar[movie_id] = similarity
    return k_most_similar


# Item-Item Collaborative Filtering
def itemCF(ratings, u, i, k):
    numerator = denominator = 0
    # find k most similar itens
    k_most_similar = k_most_similar_items(ratings, u, i, k)
    print(k_most_similar.keys())
    # similars = dictionary indexed by movie_id
    for movie_id, similarity in k_most_similar.items():
        print("MOVIE: {} RATING: {}".format(movie_id, ratings[u-1][movie_id-1]))
        numerator += similarity * ratings[u-1][movie_id-1]
        denominator += similarity
    prediction = (numerator / denominator)
    print("Rating predicted: {}".format(prediction))
    return prediction


def main():
    # get current process for performance profiling
    pid = os.getpid()
    py = psutil.Process(pid)

    # read dataset
    movies_data = pandas.read_csv("csv/movies_data.csv")
    test_data = pandas.read_csv("csv/test_data.csv")
    train_data = pandas.read_csv("csv/train_data.csv")
    users_data = pandas.read_csv("csv/users_data.csv")
    submit_file = pandas.read_csv("csv/submit_file.csv")

    # initialize our data matrix
    n_users = train_data['user_id'].max()
    n_items = movies_data['movie_id'].max()
    # unknown ratings are filled with None
    train_data_matrix = np.full((n_users, n_items), None)
    test_data_matrix = np.full((n_users, n_items), None)

    memoryUse_before = py.memory_info()[0]/2.**30
    # generate (user x movie) ratings matrix
    print("Generating user x movie ratings matrix")
    with progressbar.ProgressBar(max_value=len(train_data)) as bar:
        counter = 0
        for row in train_data.itertuples():
            user = getattr(row, "user_id") - 1
            movie = getattr(row, "movie_id") - 1
            rating = getattr(row, "rating")
            train_data_matrix[user][movie] = rating
            counter += 1
            bar.update(counter)

    # measure RAM impact
    memoryUse_after = py.memory_info()[0]/2.**30  # memory use in GB
    memoryUse = memoryUse_after - memoryUse_before
    print('~ RAM usage for matrix: {0:.3g}GB'.format(memoryUse))

    # plot initial ratings heatmap
    # sns.set()
    # sns.set_context("poster")
    # sns.heatmap(train_data_matrix, xticklabels=False, yticklabels=False, cmap='viridis', cbar_kws={"label": "rating"})
    # figure = plt.gcf()  # get current figure
    # figure.set_size_inches(8, 6)
    # # save with high DPI
    # plt.savefig("plots/initial_ratings.png", dpi = 100)

    # split training data into TRAIN and VALIDATION

    # run algorithms with TEST
    print("Generating tests dictionary...")
    print("Run ITEM-ITEM-COLLABORATIVE-FILTERING")
    for row in test_data.itertuples():
        user = getattr(row, "user_id")
        movie = getattr(row, "movie_id")
        movie_name = movies_data['title'][movie-1]
        # print("What rating would {} give for {}?".format(user, movie_name))
        # run recommendation algorithms for (u, i)
        start = timer()
        print("Predicting user {} rating for movie {}".format(user, movie_name))
        itemCF(train_data_matrix, user, movie, 50)
        end = timer()
        print("Elapsed time: {}".format(end - start))


if __name__ == '__main__':
    try:
        main()
    except:
        # colorize error output
        import re
        from sys import exc_info
        from traceback import format_exception

        RED, REV = r'\033[91m', r'\033[0m'
        err = ''.join(format_exception(*exc_info()))
        print(re.sub(r'(\w*Err\w*)', RED + r'\1' + REV, err))
