#!/bin/usr/env python3
# -*- coding: utf-8 -*-

"""
SCC0284 - Recommender Systems - 2018/1
Assignment 2
ICMC - University of Sao Paulo
Professor Marcelo Manzato
Student: Felipe Scrochio Custódio - 9442688
"""

from timeit import default_timer as timer
from termcolor import colored

import pandas
import csv
import math
import numpy as np


# read dataset
print("\nLoading dataset...")
movies_data = pandas.read_csv("csv/movies_data.csv")
users_data = pandas.read_csv("csv/users_data.csv")
train_data = pandas.read_csv("csv/train_data.csv")
test_data = pandas.read_csv("csv/test_data.csv")


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
        pass


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
