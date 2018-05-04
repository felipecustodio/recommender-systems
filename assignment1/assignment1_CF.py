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


###############################
# STOCHASTIC GRADIENT DESCENT #
###############################

def SGD(ratings, user, item):
    pass


###############################
# MATRIX DECOMPOSITION (SVD)  #
###############################

def SVD(ratings, user, item):
    u, s, v = np.linalg.svd(ratings)
    # prediction with absolute ratings
    # prediction with relative ratings


###################
# BASELINE METHOD #
###################

def global_average(ratings):
    size = 0
    total = 0
    for rating in ratings:
        if (rating != 0):
            total += rating
            size += 1
    return (total / size)


def bias_item(ratings, item, global_avg):
    item_ratings = ratings[:, item]
    bias = sum(item_ratings) - global_avg
    Ri = 0
    Ri = sum(1 for i in range(len(item_ratings)) if item_ratings[i] != 0)
    bias = bias / Ri
    return bias


def bias_user(ratings, user, global_avg):
    user_ratings = ratings[user]
    bias = 0
    Ru = 0
    for item, rating in enumerate(user_ratings):
        if (rating != 0):
            bi = bias_item(ratings, item, global_avg)
            bias += rating - global_avg - bi
            Ru += 1
    if (Ru == 0):
        # no other items were rated
        bias = global_avg
    else:
        bias = bias / Ru
    return bias


def baseline(ratings, user, item):
    u = global_average(ratings)
    bu = bias_user(ratings, user, u)
    bi = bias_item(ratings, item, u)
    rui = u + bi + bu
    return rui


########################
# PROBABILISTIC METHOD #
########################

def bayes_method(ratings, user, item):
    probabilities = [0, 0, 0, 0, 0]

    for i in range(1, 6):
        # P(Y)
        # P(item i = 1)
        # P(item i = 2)
        # P(item i = 3)
        # P(item i = 4)
        # P(item i = 5)
        pass


###########
# RF-REC #
###########

def rf_rec(ratings, user, item):
    # get all user and item ratings
    user_ratings = ratings[user]
    item_ratings = ratings[:, item]
    # initialize frequencies + 1
    frequencies_user = [1, 1, 1, 1, 1]
    frequencies_item = [1, 1, 1, 1, 1]
    rui = [0, 0, 0, 0, 0]
    # get frequency of all possible ratings
    # by user 'user' and by item 'item'
    for i in range(1, 6):
        for rating in user_ratings:
            if (rating == i):
                frequencies_user[i-1] += 1
        for rating in item_ratings:
            if (rating == i):
                frequencies_item[i-1] += 1
        # multiply rating frequency for user and item
        rui[i-1] = frequencies_user[i-1] * frequencies_item[i-1]
    # pred = arg max freq(user) x freq(item)
    prediction = rui.index(max(rui)) + 1
    return prediction


#####################################
# ITEM-ITEM COLLABORATIVE FILTERING #
#####################################

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
            # similarities[movie_id] = similarities_matrix[i][movie_id]

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


def itemCF(ratings, u, i, k):
    numerator = 0
    denominator = 0
    # find k most similar itens
    k_most_similar = k_most_similar_items(ratings, u, i, k)
    # k_most_similars = dictionary indexed by movie_id
    for movie_id, similarity in k_most_similar.items():
        numerator += similarity * ratings[u][movie_id]
        denominator += similarity
    prediction = (numerator / denominator)
    return prediction


########
# MAIN #
########

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
    for row in train_data.itertuples():
        user = getattr(row, "user_id")
        movie = getattr(row, "movie_id")
        rating = getattr(row, "rating")
        train_data_matrix[user-1][movie-1] = rating

    # split training data into TRAIN and VALIDATION

    # run algorithms with TEST
    total_time = 0
    # print("BASELINE")
    for row in test_data.itertuples():
        id = getattr(row, "id")
        user = getattr(row, "user_id")
        movie = getattr(row, "movie_id")
        movie_name = movies_data['title'][movie-1]
        # run recommendation algorithms for (u, i)
        start = timer()
        prediction = itemCF(train_data_matrix, user-1, movie-1, 20)
        print("{}, {}".format(id, prediction))
        end = timer()
        # print("Elapsed time: {}s".format(end - start))
        total_time += (end-start)
    print("TOTAL TIME: {}s".format(total_time))


if __name__ == '__main__':
    try:
        import IPython.core.ultratb
    except ImportError:
        # No IPython. Use default exception printing.
        pass
    else:
        import sys
        sys.excepthook = IPython.core.ultratb.ColorTB()
        main()
