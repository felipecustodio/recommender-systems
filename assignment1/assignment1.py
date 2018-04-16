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

import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    train_data_matrix = np.empty((n_users, n_items))
    test_data_matrix = np.empty((n_users, n_items))

    counter = 0
    print("Generating user x movie ratings matrix")
    widgets = [progressbar.Percentage(), ' ', progressbar.AnimatedMarker()]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=len(train_data))
    for row in train_data.itertuples():
        user = getattr(row, "user_id") - 1
        movie = getattr(row, "movie_id") - 1
        rating = getattr(row, "rating")
        train_data_matrix[user][movie] = rating
        counter += 1
        bar.update(counter)
    print("")

    sns.set()
    sns.set_context("poster")
    sns.heatmap(train_data_matrix, xticklabels=False, yticklabels=False)
    plt.savefig("plots/initial_ratings.png")


if __name__ == '__main__':
    main()
