#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0284 - Recommender Systems - 2018/1
Assignment 1
ICMC - University of Sao Paulo
Professor Marcelo Manzato
Student: Felipe Scrochio Custódio - 9442688
"""

import progressbar
from timeit import default_timer as timer

import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Algorithms
def user_mean_ratings(ratings, user):
    # get only existing ratings
    # filter unknown values (0)
    existing_ratings = list(filter(lambda a: a != 0, ratings[user]))
    mean = np.average(existing_ratings)
    return mean


def similarity_item(ratings, i, j):
    numerator, denominator, denominator1, denominator2 = (0, 0, 0, 0)
    # find all users that rated i and j (U)
    U = []
    # slice i and j columns
    i_ratings = ratings[:, i]
    j_ratings = ratings[:, j]
    for user_id, rating in enumerate(i_ratings):
        # if both items were rated by user, append to U
        if (i_ratings[user_id] != 0):
            if (j_ratings[user_id] != 0):
                U.append(user_id)

    # if users were found, calculate similarity
    if U:
        for user in U:
            # get user ratings for i and j
            rui = i_ratings[user]
            ruj = j_ratings[user]
            # get user globan mean
            ru = user_mean_ratings(ratings, user)
            # calculate similarity
            numerator += ((rui - ru) * (ruj - ru))
            denominator1 += (np.power((rui - ru), 2))
            denominator2 += (np.power((ruj - ru), 2))
        denominator = np.sqrt(denominator1) * np.sqrt(denominator2)
        try:
            similarity = ((float(numerator)) / (float(denominator)))
        except ZeroDivisionError:
            # division by zero probably means there was only
            # one user that rated both items i and j, rating
            # with his mean, which yields division by zero
            similarity = 0
    else:
        # no users found, set similarity to least possible
        similarity = 0
    if (similarity < 0):
        similarity = 0
    return similarity


def k_most_similar_items(ratings, u, i, k):
    # similarity list has size 'number of items'
    similarities = np.full((ratings.shape[1]), -1, dtype=float)
    # items that user 'u' rated
    user_ratings = ratings[u]

    # find all similarities with item 'i'
    # that 'u' has rated
    for movie_id, rating in enumerate(user_ratings):
        # filter items that user did not rate
        if (rating != 0):
            similarities[movie_id] = similarity_item(ratings, i, movie_id)

    # sort similarities list and get 'k' most similar
    k_biggest = similarities[np.argsort(similarities)[-k:]]
    k_biggest = k_biggest[::-1]

    # after we sort, we lose the movie_id as a position
    # we need to know which movie_id has that similarity
    # that's possible by finding the index where that
    # similarity occured in the previous 'similarity' list

    # turn list into dictionary
    k_most_similar = {}
    for similarity in k_biggest:
        # get movie_id that has similarity 'similarity'
        movie_id = (similarities.tolist()).index(similarity)
        # index similarity by movie_id
        k_most_similar[movie_id] = similarity
    return k_most_similar


# Item-Item Collaborative Filtering
def itemCF(ratings, u, i, k):
    numerator = 0
    denominator = 0
    # find k most similar itens
    k_most_similar = k_most_similar_items(ratings, u, i, k)
    print("K_most_similars:")
    print(k_most_similar)
    # k_most_similars = dictionary indexed by movie_id
    for movie_id, similarity in k_most_similar.items():
        numerator += similarity * ratings[u][movie_id]
        denominator += similarity
    prediction = (numerator / denominator)
    return prediction


def main():
    # read dataset
    movies_data = pandas.read_csv("csv/movies_data.csv")
    test_data = pandas.read_csv("csv/test_data.csv")
    train_data = pandas.read_csv("csv/train_data.csv")
    users_data = pandas.read_csv("csv/users_data.csv")
    submit_file = pandas.read_csv("csv/submit_file.csv")

    # initialize our data matrix
    n_users = train_data['user_id'].max()
    n_items = movies_data['movie_id'].max()
    # unknown ratings are filled with 0
    train_data_matrix = np.full((n_users, n_items), 0)
    test_data_matrix = np.full((n_users, n_items), 0)

    # generate (user x movie) ratings matrix
    print("Generating user x movie ratings matrix")
    with progressbar.ProgressBar(max_value=len(train_data)) as bar:
        counter = 0
        for row in train_data.itertuples():
            user = getattr(row, "user_id")
            movie = getattr(row, "movie_id")
            rating = getattr(row, "rating")
            train_data_matrix[user-1][movie-1] = rating
            counter += 1
            bar.update(counter)

    # split training data into TRAIN and VALIDATION

    # run algorithms with TEST
    total_time = 0
    print("Run ITEM-ITEM-COLLABORATIVE-FILTERING")
    for row in test_data.itertuples():
        user = getattr(row, "user_id")
        movie = getattr(row, "movie_id")
        movie_name = movies_data['title'][movie-1]
        # run recommendation algorithms for (u, i)
        start = timer()
        prediction = itemCF(train_data_matrix, user-1, movie-1, 20)
        print("pred({},{}) = {}".format(user, movie_name, prediction))
        end = timer()
        print("Elapsed time: {}s".format(end - start))
        total_time += (end-start)


if __name__ == '__main__':
    try:
        import IPython.core.ultratb
    except ImportError:
        # No IPython. Use default exception printing.
        pass
    else:
        import sys
        import textwrap
        sys.excepthook = IPython.core.ultratb.ColorTB()
        main()
