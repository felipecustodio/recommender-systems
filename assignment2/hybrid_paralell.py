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

import fbc_knn_users
import fbc_knn_genres
import fbc_knn_reviews

"""

"""

algorithm = "hybrid_monolithic"
