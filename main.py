# sys.argv[1]
import sys
import numpy as np


def naive_gaussian(coeffs, consts):
    forward_elimination(coeffs, consts)


def forward_elimination(coeffs, consts):
    for k in range(col - 1):
        for i in range(k+1, rows):
            coeffs[i, :] = (coeffs[i, k]/coeffs[k, k]) * coeffs[k,:] - coeffs[i, :]
    print(coeffs)
    backward_elimination(coeffs, consts)


def backward_elimination(coeffs, consts):
    x =np.zeros(col-1)
    for i in np.arange(rows-1, -1, -1):
        x[i] = (coeffs[i, -1] - coeffs[i, 0:col-1]@x)/coeffs[i, i]
    print(x)


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
        content.append(line.split(" "))


    # cleans up data from file even more by getting rid of any "\n" found in the arrays
    content = [[s.rstrip('\n') for s in nested if not s.isspace()] for nested in content]
    content = [nested for nested in content if nested]

    # declare and initialize array that we will start using as coefficients
    coefficient = [[0 for i in range(n)] for j in range(m)]
    for i in range(m):
        for j in range(n):
            coefficient[i][j] = float(content[i][j])


    # declare and initialize array that we will start using as constants
    constant_array = [0 for i in range(n)]
    for j in range(n):
        #constant_array[j] = float(content[n][j])
        last_column = coefficient[j]
        last_column.append(float(content[n][j]))
    # make constant array into a vector
    constant = np.array(constant_array)
    matrix = coefficient[:-1]
    print(len(matrix))
    print("array", matrix)



    rows = np.shape(matrix)[0]
    col = np.shape(matrix)[1]



    coefficient = np.array(matrix, dtype=float)
    naive_gaussian(coefficient, constant)

