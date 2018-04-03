import numpy as np
import matplotlib.pyplot as pp

movies_data = open("movies_data.csv")

# create dictionary of movies
movies = {}
for line in movies_data:
    movies[line.split(",")[0]] = line.split(",")[1]  # movies[id] = movie name

ratings = {}
train_data = open("train_data.csv", "r")
for line in train_data:
    pass
