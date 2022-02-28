# sys.argv[1]
import timeit

print(timeit.Timer('for i in range(10): oct(i)').timeit())
import sys
import numpy as np


def naive_gaussian(matrix):
    forward_elimination(matrix)


def forward_elimination(matrix):
    for k in range(col - 1):  # pivot row
        for i in range(k + 1, rows):  # iterate through the rows
            # "[variable, :]" means all columns in that row will be affected
            matrix[i, :] = (matrix[i, k] / matrix[k, k]) * matrix[k, :] - matrix[i, :]  # row subtraction then
            # multiplication and assigning it to the current row
    print(matrix)
    # call function to do backward elimination
    backward_elimination(matrix)


def backward_elimination(matrix):
    # initialize new array for the answer
    x = np.zeros(col - 1)

    # iterate backwards using numpy because it's faster
    for i in np.arange(rows - 1, -1, -1):  # using rows instead of columns because matrix changed from 5x4 to 4x5
        # x[i] = Bi minus dot product of the current row and the updated x array divided by Aii
        # the dot product returns an int or float
        x[i] = (matrix[i, -1] - np.dot(matrix[i, 0:col - 1], x)) / matrix[i, i]

    print("answer", x)

    # print(x)


with open("test2.txt") as filename:
    # declare array we're going to get from the file
    content = []

    # reads the first line in the file that has the number of the coefficients
    n = int(filename.readline())
    # number of rows due to the extra array for the constants
    m = n + 1
    # reads the rest of the lines within the file
    lines = filename.readlines()

    # takes each line read from file and inputs each number in the line into an element into the file content array
    for line in lines:
        content.append(line.split())  # split ignores all unnecessary characters like \n or whitespace

    # declare and initialize array that we will start using and operating
    constants_in_row_matrix = [[0 for i in range(n)] for j in range(m)]

    # formally assign file matrix into a 2D array
    for i in range(m):
        for j in range(n):
            constants_in_row_matrix[i][j] = float(content[i][j])  # change from string to float
    print("og matrix", constants_in_row_matrix)
    # declare and initialize array that we will place the constants in
    #constant_array = [0 for i in range(n)]
    for j in range(n):
        #constant_array[j] = float(content[n - 1][j])
        last_column = constants_in_row_matrix[j]
        print("current last column", last_column)
        last_column.append(float(content[n][j]))
    print("appended column matrix", constants_in_row_matrix)
    print("last column", last_column)
    # make constant array into a vector
    #constant = np.array(constant_array)
    matrix = constants_in_row_matrix[:-1]
    print(len(constants_in_row_matrix))
    # print("array", matrix)

    rows = np.shape(matrix)[0]
    col = np.shape(matrix)[1]

    # rows = np.shape(coefficient)[0]
    # col = np.shape(coefficient)[1]

    matrix = np.array(matrix, dtype=float)
    naive_gaussian(matrix)
print(timeit.Timer('for i in range(10): oct(i)').timeit())
