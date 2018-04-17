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

import time
import progressbar

import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    train_data_matrix = np.empty((n_users, n_items))
    test_data_matrix = np.empty((n_users, n_items))

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
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB
    print('RAM usage: {0:.3g}GB'.format(memoryUse))

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
    for row in test_data.itertuples():
        user = getattr(row, "user_id")
        movie = getattr(row, "movie_id")
        movie_name = movies_data['title'][movie-1]
        print("pred(user: {}, movie:{} movie_id: {})".format(user, movie_name, movie))

if __name__ == '__main__':
    main()
