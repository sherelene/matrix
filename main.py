# sys.argv[1]
import numpy as np
from decimal import *
import sys, getopt
import argparse
from argparse import ArgumentParser
import os


def forward_elimination(matrix, row, col):
    for k in range(col - 1):  # pivot row
        for i in range(k + 1, row):  # iterate through the rows
            # "[variable, :]" means all columns in that row will be affected
            matrix[i, :] = (matrix[i, k] / matrix[k, k]) * matrix[k, :] - matrix[i, :]  # row subtraction then
            # multiplication and assigning it to the current row

    # call function to do backward elimination
    backward_elimination(matrix, row, col)


def backward_elimination(matrix, row, col):
    # initialize new array to store the solved coefficients
    solution = np.zeros(col - 1, dtype=np.dtype(Decimal))

    # iterate backwards for back substitution using numpy because it's faster
    for i in np.arange(row - 1, -1, -1):  # using rows instead of columns because matrix changed from 5x4 to 4x5
        # x[i] = Bi minus dot product of the current row and the updated x array divided by Aii
        # the dot product returns an int or float
        solution[i] = (matrix[i, -1] - np.dot(matrix[i, 0:col - 1], solution)) / matrix[i, i]
        # added this block of code because one test result was returning -0 instead of 0
        if solution[i] == -0:
            solution[i] = 0

    print("Naive solution:\n", solution)
    new_lst = (' '.join(str(element) for element in solution))  # removes brackets for the .sol file as per
    # instructions in assignment
    print_to_file(new_lst)


def spp_gaussian(coefficient_matrix, constants_matrix, n):
    spp_solution = np.zeros(n, dtype=np.dtype(Decimal))
    index_vector = [i for i in range(n)]

    spp_forward_elimination(coefficient_matrix, constants_matrix, index_vector, n)
    spp_backward_elimination(coefficient_matrix, constants_matrix, index_vector, spp_solution, n)


def spp_forward_elimination(coefficient_matrix, constants_matrix, index_vector, n):
    # initialising vector of scaling factors
    scaling_vector = [0 for i in range(n)]
    for i in range(n):
        scalar_max = 0
        for j in range(n):
            scalar_max = max(scalar_max, np.abs(coefficient_matrix[i, j]))  # finds coefficient with greatest value
        # for each row
        scaling_vector[i] = scalar_max

    # pivot row
    for k in range(n - 1):
        ratio_max = 0
        max_index = k
        for i in range(k, n):
            ratio = np.abs(coefficient_matrix[index_vector[i], k]) / scaling_vector[index_vector[i]]
            if ratio > ratio_max:
                ratio_max = ratio
                max_index = i
        index_vector[max_index], index_vector[k] = index_vector[k], index_vector[max_index] #  scaping of index

        for i in range(k + 1, n):
            multiplier = coefficient_matrix[index_vector[i], k] / coefficient_matrix[index_vector[k], k]
            for j in range(k, n):
                coefficient_matrix[index_vector[i], j] = coefficient_matrix[index_vector[i], j] - multiplier * \
                                                         coefficient_matrix[index_vector[k], j]
            constants_matrix[index_vector[i]] = constants_matrix[index_vector[i]] - multiplier * constants_matrix[
                index_vector[k]]


def spp_backward_elimination(coefficient_matrix, constants_matrix, index_vector, spp_solution, n):
    for i in range(n - 1, -1, -1):
        summation = constants_matrix[index_vector[i]]
        for j in range(i + 1, n):
            summation = summation - coefficient_matrix[index_vector[i], j] * spp_solution[j]
        spp_solution[i] = summation / coefficient_matrix[index_vector[i], i]

    print("scaled partial pivoting solution:\n", spp_solution)
    new_lst = (' '.join(str(element) for element in spp_solution))  # removes brackets for the .sol file as per
    # instructions in assignment
    print_to_file(new_lst)


def print_to_file(solution):
    input_filename = args.filename
    output_filename = os.path.splitext(input_filename)[0] + ".sol"
    with open(output_filename, "w") as external_file:
        print(solution, file=external_file)
        external_file.close()
    print("\nThis solution has been placed in an output file named {}".format(output_filename))


def main(args):
    with open(args.filename) as file:
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

    if args.spp:
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

        # call method to start naive_gaussian algorithm
        forward_elimination(proper_matrix, row, col)


if __name__ == "__main__":
    # start of getting arguments from command line
    parser = argparse.ArgumentParser(
        description='gaussian algorithms with naive as default and scaled partial pivoting '
                    'as an optional flag written as: python3 gaussian <optional-flag> '
                    '<filename>')
    parser.add_argument('-spp', '--spp', action="store_true", help="calls spp algorithm")
    parser.add_argument("filename", help="stores filename")
    args = parser.parse_args()  # stores all the arguments int the commandline

    main(args)