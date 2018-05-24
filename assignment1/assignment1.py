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
from termcolor import colored

import pandas
import csv
import math
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns


###############################
# STOCHASTIC GRADIENT DESCENT #
###############################
def SGD(ratings, user, item):
    pass


###############################
#             SVD             #
###############################
def SVD_prediction(P, S, Q, user, movie, k):
    prediction = 0
    for i in range(k):
        prediction += P[user][i] * S[i] * Q[i][movie - 1]
    return prediction

def SVD(ratings):
    # matrix decomposition
    P, S, Q = np.linalg.svd(ratings)
    return P, S, Q


##############################
#        BASELINE METHOD     #
##############################
def global_average(ratings):
    size = 0
    total = 0
    for row in ratings:
        for rating in row:
            if (rating != 0):
                total += rating
                size += 1
    return (total / size)


def bias_item(ratings, item, global_avg):
    bias = 0
    Ri = 0
    for i in range(len(ratings)):
        bias += ratings[i][item] - global_avg
        Ri += 1
    if (Ri == 0):
        # no other ratings for this item
        bias = global_avg
    else:
        bias = bias / abs(Ri)
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
        bias = bias / abs(Ru)
    return bias


def baseline(ratings, user, item, global_avg):
    u = global_avg
    bu = bias_user(ratings, user, u)
    bi = bias_item(ratings, item, u)
    rui = u + bi + bu
    return rui


#########################
# PROBABILISTIC (BAYES) #
#########################
def bayes_method(ratings, user, item):
    # all ratings for given user
    user_ratings = ratings[user]
    # all ratings for given item
    item_ratings = ratings[:, item]

    # final probability
    probability = [0, 0, 0, 0, 0]

    # hypothesis:
    # probability for rating (1-5)
    probability_of_rating = [0, 0, 0, 0, 0]

    # ratings frequency for given item
    ratings_frequencies_item = [0, 0, 0, 0, 0]

    # calculate frequency for ratings (for given item)
    for rating in item_ratings:
        # rating is in range (1-5), represented by indexes (0-4)
        ratings_frequencies_item[rating - 1] += 1

    # calculate probability of each rating for given item
    for i in range(5):
        probability_of_rating[i] = ratings_frequencies_item[i] / sum(ratings_frequencies_item)

    # for every movie that user has rated
    # calculate the probability of each rating
    conditional_probabilities = np.zeros((len(user_ratings), 5))
    for movie, rating in enumerate(user_ratings):
        item_ratings = ratings[:, movie]
        for i in range(5):
            total_frequency = 0
            # frequency that rating shows for this movie
            for rating in item_ratings:
                if (rating == (i+1)):
                    total_frequency += 1
            # conditional probability = frequency of rating for
            # desired item i / frequency of rating for item j
            if (total_frequency > 0):
                conditional_probabilities[movie][i] = ratings_frequencies_item[i] / total_frequency
            else:
                conditional_probabilities[movie][i] = 0

    # probability of rating i = probability of rating i
    # for desired item * product of conditional probabilities for rating i
    for i in range(5):
        productory = np.prod(conditional_probabilities[:, i])
        probability[i] = probability_of_rating[i] * productory

    prediction = np.argmax(probability) +  1
    return prediction


###########
# RF-REC #
###########
def RF_Rec(ratings, user, item):
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
        # rating frequency = frequency that user gave
        # that rating * frequency that item was given that rating
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
    user_ratings = ratings[user]
    total = 0
    sum = 0
    for rating in user_ratings:
        if (rating != 0):
            sum += rating
            total += 1
    mean = (sum / total)
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
        # round up to avoid floating errors
        similarity = 0
    return similarity


def k_most_similar_items(ratings, u, i, k):
    # similarity list has size 'number of items'
    similarities = np.full((ratings.shape[1]), 0, dtype=float)
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


def scale_range(input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input


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
    # normalize if values didn't reach required range
    if (prediction < 1 or prediction > 5):
        prediction = scale_range(prediction, 1, 5)
    # if it fails terribly, return the average for that item
    if (math.isnan(prediction)):
        prediction = np.average(ratings[:, i])
    return prediction


########
# MAIN #
########
def rmse(rating, prediction):
    rmse = sqrt(((rating - prediction) ** 2).mean(axis=None))
    return rmse


def error_check(prediction, id):
    if (prediction < 1 or prediction > 5 or math.isnan(prediction)):
        text = colored('ERROR: ', 'red', attrs=['reverse', 'blink'])
        print(text + "prediction {} at position {}".format(prediction, id))


def main():
    text = colored('Recommender Systems - Assignment 1', 'white', attrs=['reverse', 'blink'])
    print(text)

    # choose algorithm
    print("\nChoose recommendation algorithm: ")
    print("\t 1 - Item-Item Collaborative Filtering")
    print("\t 2 - Baseline")
    print("\t 3 - RF-Rec")
    print("\t 4 - Bayes")
    print("\t 5 - SVD")
    print("\t 6 - SGD")
    print("\t 7 - Skip to performance report")
    print("::: ", end='')
    method = int(input())

    # read dataset
    print("\nLoading dataset...")
    movies_data = pandas.read_csv("csv/movies_data.csv")
    test_data = pandas.read_csv("csv/test_data.csv")
    train_data = pandas.read_csv("csv/train_data.csv")

    # initialize our data matrix (0 = unknown rating)
    n_users = train_data['user_id'].max()
    n_items = movies_data['movie_id'].max()
    ratings = np.full((n_users, n_items), 0)

    # generate (user x movie) ratings matrix
    print("Generating user x movie ratings matrix...")
    for row in train_data.itertuples():
        user = getattr(row, "user_id")
        movie = getattr(row, "movie_id")
        rating = getattr(row, "rating")
        ratings[user-1][movie-1] = rating

    print("Calculating global average for ratings...")
    global_avg = global_average(ratings)

    print("Calculating SVD...")
    P, S, Q = SVD(ratings)

    # name results CSV
    if (method == 1):
        algorithm = "itemCF"
    elif (method == 2):
        algorithm = "baseline"
    elif (method == 3):
        algorithm = "rfrec"
    elif (method == 4):
        algorithm = "bayes"
    elif (method == 5):
        algorithm = "svd"
    elif (method == 6):
        algorithm = "sgd"
    elif (method == 7):
        algorithm = "null"

    # write results CSV header
    results_csv = open('results/' + algorithm + '.csv', 'w', newline='')
    results_writer = csv.writer(results_csv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['id', 'rating'])

    # run tests and write to results csv
    counter = 0
    times = []
    print("Calculating predictions...")
    with progressbar.ProgressBar(max_value=3970) as bar:
        for row in test_data.itertuples():
            id = getattr(row, "id")
            user = getattr(row, "user_id")
            movie = getattr(row, "movie_id")
            # run chosen recommendation algorithm for (u, i)
            start = timer()
            if (method == 1):
                prediction = itemCF(ratings, user-1, movie-1, 20)
                error_check(prediction, id)
            elif (method == 2):
                prediction = baseline(ratings, user-1, movie-1, global_avg)
                error_check(prediction, id)
            elif (method == 3):
                prediction = RF_Rec(ratings, user-1, movie-1)
                error_check(prediction, id)
            elif (method == 4):
                prediction = bayes_method(ratings, user-1, movie-1)
                error_check(prediction, id)
            elif (method == 5):
                prediction = SVD_prediction(P, S, Q, user-1, movie-1, 5)
                error_check(prediction, id)
            elif (method == 6):
                prediction = SGD(ratings, user-1, movie-1)
                error_check(prediction, id)
            elif (method == 7):
                pass

            end = timer()
            time_elapsed = end - start
            times.append(time_elapsed)
            # write results to csv
            if (method != 7):
                results_writer.writerow([id, prediction])
            counter += 1
            bar.update(counter)

    print("Plotting time elapsed...")
    sns.set()
    # plot time for each iteration
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

    print("Code profiling...")
    # code profiling
    # plot performance: time elapsed for each prediction and rmse
    times = {}
    times["itemcf"] = []
    times["baseline"] = []
    times["rfrec"] = []
    times["bayes"] = []
    times["svd"] = []
    times["sgd"] = []

    rmses = {}
    rmses["itemcf"] = []
    rmses["baseline"] = []
    rmses["rfrec"] = []
    rmses["bayes"] = []
    rmses["svd"] = []
    rmses["sgd"] = []

    counter = 0
    print("Split train data size: ", end='')
    split_size = int(input())
    train_split = train_data.tail(split_size)
    with progressbar.ProgressBar(max_value=split_size) as bar:
        for row in train_split.itertuples():
            # id = getattr(row, "id")
            user = getattr(row, "user_id")
            movie = getattr(row, "movie_id")
            rating = getattr(row, "rating")

            start = timer()
            prediction = itemCF(ratings, user-1, movie-1, 20)
            error_check(prediction, counter)
            end = timer()
            time_elapsed = end - start
            times["itemcf"].append(time_elapsed)
            rmses["itemcf"].append(rmse(rating, prediction))

            start = timer()
            prediction = baseline(ratings, user-1, movie-1, global_avg)
            error_check(prediction, counter)
            end = timer()
            time_elapsed = end - start
            times["baseline"].append(time_elapsed)
            rmses["baseline"].append(rmse(rating, prediction))

            start = timer()
            prediction = RF_Rec(ratings, user-1, movie-1)
            error_check(prediction, counter)
            end = timer()
            time_elapsed = end - start
            times["rfrec"].append(time_elapsed)
            rmses["rfrec"].append(rmse(rating, prediction))

            start = timer()
            prediction = bayes_method(ratings, user-1, movie-1)
            end = timer()
            time_elapsed = end - start
            times["bayes"].append(time_elapsed)
            rmses["bayes"].append(rmse(rating, prediction))

            start = timer()
            prediction = SVD_prediction(P, S, Q, user-1, movie-1, 5)
            end = timer()
            time_elapsed = end - start
            times["svd"].append(time_elapsed)
            rmses["svd"].append(rmse(rating, prediction))

            # start = timer()
            # prediction = SGD(ratings, user-1, movie-1)
            # end = timer()
            # time_elapsed = end - start
            # times["sgd"].append(time_elapsed)
            # rmses["sgd"].append(rmse(rating, prediction))

            counter += 1
            bar.update(counter)


    # colors (FLATUI)
    print("Plotting time elapsed...")
    # plot time for each iteration
    plt.scatter(range(len(times["itemcf"])), times["itemcf"], s=4, c="#f1c40f", label="item-item CF")
    plt.scatter(range(len(times["baseline"])), times["baseline"], s=4, c="#c0392b", label="Baseline")
    plt.scatter(range(len(times["rfrec"])), times["rfrec"], s=4, c="#2c3e50", label="RF-Rec")
    plt.scatter(range(len(times["bayes"])), times["bayes"], s=4, c="#2980b9", label="Bayes")
    plt.scatter(range(len(times["svd"])), times["svd"], s=4, c="#27ae60", label="SVD")
    # plt.scatter(range(len(times["sgd"])), times["sgd"], s=4, c="#bdc3c7", label="SGD")

    plt.legend()
    plt.xlabel("Test case (ID)")
    plt.ylabel("Time elapsed (seconds)")
    plt.title("Time elapsed")

    # get current figure
    figure = plt.gcf()
    # # 800 x 600
    figure.set_size_inches(8, 6)
    # # save with high DPI
    plt.savefig("plots/time_elapsed.png", dpi=100)
    plt.clf()

    print("Plotting RMSE...")
    # sns.set()
    # plot time for each iteration
    plt.scatter(range(len(rmses["itemcf"])), rmses["itemcf"], s=4, c="#f1c40f", label="item-item CF")
    plt.scatter(range(len(rmses["baseline"])), rmses["baseline"], s=4, c="#c0392b", label="Baseline")
    plt.scatter(range(len(rmses["rfrec"])), rmses["rfrec"], s=4, c="#2c3e50", label="RF-Rec")
    plt.scatter(range(len(rmses["bayes"])), rmses["bayes"], s=4, c="#2980b9", label="Bayes")
    plt.scatter(range(len(rmses["svd"])), rmses["svd"], s=4, c="#27ae60", label="SVD")
    # plt.scatter(range(len(rmses["sgd"])), rmses["sgd"], s=4, c="#bdc3c7", label="SGD")

    plt.legend()
    plt.xlabel("Test case (ID)")
    plt.ylabel("RMSE")
    plt.title("RMSE")

    figure = plt.gcf()  # get current figure
    # # 800 x 600
    figure.set_size_inches(8, 6)
    # # save with high DPI
    plt.savefig("plots/rmse.png", dpi=100)


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
