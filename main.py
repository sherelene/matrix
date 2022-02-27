# sys.argv[1]
import sys
import numpy as np


def naive_gaussian(coeffs, consts):
    num_of_coeff = len(coeffs)
    solution = [0 for i in range(n)]
    forward_elimination(coeffs, consts)
    backward_elimination(coeffs, consts)


def forward_elimination(coeffs, consts):
    num_of_coeff = len(coeffs)

    for k in range(col - 1):
        for i in range(k+1, rows):
            coeffs[i:] = coeffs[i:] - (coeffs[i][k]/coeffs[k][k]) * coeffs[i:]
    print(coeffs)



    #print("new coefficients", coefficient)
    #print("new constants", consts)


def backward_elimination(coeffs, consts):
    length = len(coeffs)-1
    x =np.zeros(length)
    #for i in np.arange(rows-1, -1, -1):
        #x[i] = (coeffs[i][-1] - coeffs[i, 0:col-1]@x)/coeffs[i][i]
    #print(x)


with open("practice_input.txt") as filename:
    # declare array we're going to get from the file
    content = []

    # reads the first line in the file that has the size of the matrix
    n = int(filename.readline())
    m = n - 1
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
    coefficient = [[0 for i in range(m)] for j in range(n)]
    for i in range(n):
        for j in range(m):
            coefficient[i][j] = float(content[i][j])


    # declare and initialize array that we will start using as constants
    constant_array = [0 for i in range(n)]
    for j in range(m):
        #constant_array[j] = float(content[n][j])
        last_column= coefficient[j]
        last_column.append(float(content[m][j]))
    # make constant array into a vector
    constant = np.array(constant_array)
   # coefficient = np.append(coefficient, constant, axis=1)
    print(coefficient)



    rows = np.shape(last_column)[0]
    col = np.shape(last_column)[1]
   # print(content)
    #print(type(coefficient[0][0]))
    #print(constant_array)

    print(n)
    print(len(coefficient))

    coefficient = np.array(coefficient, dtype=float)
    naive_gaussian(coefficient, constant)

