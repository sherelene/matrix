# sys.argv[1]
import timeit

print(timeit.Timer('for i in range(10): oct(i)').timeit())
import sys
import numpy as np
import pandas as pd
from decimal import *
import re


def naive_gaussian(coeffs, consts):
    forward_elimination(coeffs, consts)


def forward_elimination(coeffs, consts):
    for k in range(col - 1): #pivot element
        for i in range(k + 1, rows): #iterate through the rows
            coeffs[i, :] = (coeffs[i, k] / coeffs[k, k]) * coeffs[k, :] - coeffs[i, :]
    print(coeffs)
    backward_elimination(coeffs, consts)


def backward_elimination(coeffs, const):
    # x = np.zeros(n, float)

    # x[col-1] = b[col-1] / A[col-1, col-1]
    x = np.zeros(col - 1)
    for i in np.arange(rows - 1, -1, -1):
        x[i] = (coeffs[i, -1] - np.dot(coeffs[i, 0:col - 1], x)) / coeffs[i, i]
    print(x)

    print("answer", x)

    # print(x)


with open("practice_input.txt") as filename:
    # declare array we're going to get from the file
    content = []

    # reads the first line in the file that has the size of the matrix
    n = int(filename.readline())
    m = n + 1
    # reads the rest of the lines within the file
    lines = filename.readlines()

    # takes each line read from file and inputs each number into an element into
    # the file content array

    for line in lines:
        content.append(line.split())

    # declare and initialize array that we will start using as coefficients
    coefficient = [[0 for i in range(n)] for j in range(m)]
    for i in range(m):
        for j in range(n):
            coefficient[i][j] = float(content[i][j])

    # declare and initialize array that we will start using as constants
    constant_array = [0 for i in range(n)]
    for j in range(n):
        constant_array[j] = float(content[n - 1][j])
        last_column = coefficient[j]
        last_column.append(float(content[n][j]))
    # make constant array into a vector
    constant = np.array(constant_array)
    matrix = coefficient[:-1]
    print(len(coefficient))
    # print("array", matrix)

    rows = np.shape(matrix)[0]
    col = np.shape(matrix)[1]

    # rows = np.shape(coefficient)[0]
    # col = np.shape(coefficient)[1]

    matrix = np.array(matrix, dtype=float)
    naive_gaussian(matrix, constant)
print(timeit.Timer('for i in range(10): oct(i)').timeit())