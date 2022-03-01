# sys.argv[1]
import numpy as np
from decimal import *

import sys


def forward_elimination(matrix, row, col):
    for k in range(col - 1):  # pivot row
        for i in range(k + 1, row):  # iterate through the rows
            # "[variable, :]" means all columns in that row will be affected
            matrix[i, :] = (matrix[i, k] / matrix[k, k]) * matrix[k, :] - matrix[i, :]  # row subtraction then
            # multiplication and assigning it to the current row
    print(matrix)
    # call function to do backward elimination
    backward_elimination(matrix, row, col)


def backward_elimination(matrix, row, col):
    # initialize new array to store the solved coefficients
    x = np.zeros(col - 1, dtype=np.dtype(Decimal))

    # iterate backwards for back substitution using numpy because it's faster
    for i in np.arange(row - 1, -1, -1):  # using rows instead of columns because matrix changed from 5x4 to 4x5
        # x[i] = Bi minus dot product of the current row and the updated x array divided by Aii
        # the dot product returns an int or float
        x[i] = (matrix[i, -1] - np.dot(matrix[i, 0:col - 1], x)) / matrix[i, i]
        # added this block of code because one test result was returning -0 instead of 0
        if x[i] == -0:
            x[i] = 0

    print("answer", x)

def spp_gaussian(coefficient_matrix, constants_matrix, n):
    solution_vector = [i for i in range(n)]
    index_vector = [i for i in range(n)]

    spp_forward_elimination(coefficient_matrix, constants_matrix, index_vector, n)

def spp_forward_elimination(coefficient_matrix, constants_matrix, index_vector, n):
    # initialising vector of scaling factors
    #scaling_vector = np.zeros(n, dtype=np.dtype(Decimal))
    print("og", coefficient_matrix)
    scaling_vector = [0 for i in range(n)]
    for i in range(n):
        scalar_max = 0
        for j in range(n):
            scalar_max = max(scalar_max, np.abs(coefficient_matrix[i, j]))
        scaling_vector[i] = scalar_max   # finds coefficient with greatest value for each row
    print(scalar_max)
    print(scaling_vector)

    # pivot row
    for k in range(n - 1):
        ratio_max = 0
        max_index = k
        for i in range(k, n):
            ratio = np.abs(coefficient_matrix[index_vector[i], k]) / scaling_vector[index_vector[i]]
            print("ratio", ratio)
            if ratio > ratio_max:
                ratio_max = ratio
                max_index = i
        index_vector[max_index], index_vector[k] = index_vector[k], index_vector[max_index]
        print(index_vector)
        print(max_index)

        for i in range(k+1, n):
            multiplier = coefficient_matrix[index_vector[i], k] / coefficient_matrix[index_vector[k], k]
            for j in range(k, n):
                coefficient_matrix[index_vector[i], j] = coefficient_matrix[index_vector[i], j] - multiplier * \
                                                  coefficient_matrix[index_vector[k], j]
            constants_matrix[index_vector[i]] = constants_matrix[index_vector[i]] - multiplier * constants_matrix[
                index_vector[k]]
    print("new", coefficient_matrix)



# def spp_backward_elimination():


# def spp_scaling_pivot():

filename = "test2.txt"
with open("test2.txt") as file:
    # declare array we're going to get from the file
    content = []

    # reads the first line in the file that has the number of the coefficients
    n = int(file.readline())
    # number of rows due to the extra array containing the constants
    m = n + 1
    # reads the rest of the lines within the file
    lines = file.readlines()

    # takes each line read from file and inputs each number in the line into an element into the file content array
    for line in lines:
        content.append(line.split())  # split ignores all unnecessary characters like \n or whitespace

    # declare and initialize array that we will start using and operating on
    constants_in_row_matrix = [[0 for i in range(n)] for j in range(m)]
    # formally assign matrix from file into a 2D array
    for i in range(m):
        for j in range(n):
            constants_in_row_matrix[i][j] = float(content[i][j])  # change from string to float

    print("og", constants_in_row_matrix)

argument = True

if argument:
    # separate file matrix into a coefficient matrix and a constants matrix for easy operations
    coefficient_matrix = np.array(constants_in_row_matrix[:-1])
    constants_matrix = np.array(constants_in_row_matrix[-1])

    # call method to start scaled partial pivoting algorithm
    spp_gaussian(coefficient_matrix, constants_matrix, n)

else:
    for j in range(n):
        # assigning an array to another variable passes by reference so whatever changes you make on that variable
        # will reflect on the original array
        last_column = constants_in_row_matrix[j]
        # make constants that are in the last row into the last column
        last_column.append(constants_in_row_matrix[n][j])

    # remove last row from constants_in_row_matrix for a "proper" matrix that has the constants in the last column
    proper_matrix = constants_in_row_matrix[:-1]

    # make variables for new number of rows and columns
    row = np.shape(proper_matrix)[0]
    col = np.shape(proper_matrix)[1]

    # turn our matrix into a numpy matrix array for numpy perks
    proper_matrix = np.array(proper_matrix, dtype=np.dtype(Decimal))
    print(proper_matrix)
    # call method to start naive_gaussian algorithm
    forward_elimination(proper_matrix, row, col)
