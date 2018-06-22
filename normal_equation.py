import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)
data = pd.read_csv("ex1data2.csv")
Y = data["price"]
line, col = np.shape(data)

Y = np.reshape(Y, (np.size(Y), 1))
X = [np.insert(row, 0, 1) for row in data.drop(["price"], axis=1).values]
X = np.reshape(X, (line, col))
XT = np.transpose(X)
print(X)
print(XT)
theta = [[0.0] * col]
theta = np.reshape(theta, (col, 1))
size = np.size(theta)

def normal_equation(X, XT, Y):
    ret = XT.dot(X)
    print(ret)
    ret = np.linalg.inv(ret)
    ret = ret.dot(XT)
    ret = ret.dot(Y)
    return (ret)

theta = normal_equation(X, XT, Y)
print(theta[0], "\n", theta[1], " price\n", theta[2], " nb_bedrooms\n")
size_meters = 852
nb_rooms = 2
result = theta[0] + theta[1] * size_meters + theta[2] * nb_rooms
print(result)
