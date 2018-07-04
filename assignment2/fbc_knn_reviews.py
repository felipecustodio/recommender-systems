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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

"""
FBC-kNN
Using movie reviews
"""

algorithm = "FBC-kNN-reviews"


def similarity(movie_ids, movie_reviews, movies_data):
    # for each movie:
        # generate a TF/IDF vector of the terms in the movie reviews
        # calculate the cosine similarity of each movie's TF/IDF vector with every other movie's TF/IDF vector

    # corpus consists of all reviews
    corpus = ["" for x in range(len(movies_data))]

    # concatenate movie reviews to generate corpus
    print("\nGenerating reviews corpus...")
    for row in movie_reviews.itertuples():
        movie = getattr(row, "movie_id")
        review = getattr(row, "text")
        corpus[movie-1] += str(review)

    # initialize vectorizer and matrix for each movie review
    print("Generating TF-IDF vectorizer...")
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
    tfidf_matrix = tf.fit_transform([review for index, review in enumerate(corpus)])

    # calculate vector of similarities for each of the movies from the first recommender
    # recommend the most similar for each one for the user

    top_n = 1
    most_similar = []

    print("Finding most similar movies...")
    for movie in movie_ids:
        cosine_similarities = linear_kernel(tfidf_matrix[movie:movie+1], tfidf_matrix).flatten()
        related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != movie]
        most_similar.append([(movie, cosine_similarities[movie]) for movie in related_docs_indices][0:top_n])

    print("\n\nRecommended movies for you:")
    # for movie in most_similar:
    #     print("* " + movies_data['title'][movie[0][0]-1])

    return most_similar

def FBC_kNN_reviews(user, item, similarity, data):

    user = data.loc[data['user_id'] == user]
    similar_itens = np.copy(similarity[item - 1])
    sorted_indexes = np.argsort(similar_itens)[-1:-neighbors:-1]

    sum_a = 0
    sum_b = 0

    for index in sorted_indexes:

        rated_movie = user.loc[user['movie_id'] == (index + 1)]

        if rated_movie.empty == False:
            sum_a += similar_itens[index] * rated_movie.rating.values[0]
            sum_b += similar_itens[index]

    if (sum_a == 0) or (sum_b == 0):
        return 0
    else:
        return (sum_a / sum_b)


# read dataset
movies_data = pandas.read_csv("csv/movies_data.csv")
test_data = pandas.read_csv("csv/test_data.csv")
reviews_data = pandas.read_csv("csv/movie_reviews.csv")

write results CSV header
results_csv = open('results/' + algorithm + '.csv', 'w', newline='')
results_writer = csv.writer(results_csv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
results_writer.writerow(['id', 'rating'])

movies = np.sort(movies_data.movie_id.unique())
similarity_matrix = similarity(movies, reviews_data, movies_data)

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
        prediction = FBC_kNN_reviews(user, movie, similarity_matrix, train_data)

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
