# LR Exercise using insurance.csv 

import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd

# import csv datafile
data_insurance = pd.read_csv(os.path.join("Dataset", "insurance.csv"), delimiter=",")

# one variable LR
Age, Cost = data_insurance["age"].iloc[:], data_insurance["charges"].iloc[:]
m = Cost.size
n = Age.size

"""
Visualisation of the input data
"""
def plotData(X, y):
    fig = plt.figure() 
    plt.plot(X, y, 'ro')
    plt.ylabel("Insurance Cost")
    plt.xlabel("Age")


plotData(Age, Cost)
# plt.show()


"""
Perform LR Analysis
"""

# Cost function J as a function of certain theta - linear function
def computeCost(X, y, theta):
    h = np.dot(X, theta) # initial cost predictions using theta
    J = (1/(2*m)) * np.sum(np.square(h - y))

    print(h)

    return J 

Age_stack = np.stack([np.ones(n), Age], axis=1) # stack for one-dimensional dataset of Age


J = computeCost(Age_stack, Cost, theta=np.array([0, 0]))

# Application of gradient descent to retrieve most optimal theta combination
def gradientDescent(X, y, theta, alpha, num_iters):
    theta = theta.copy() # initial start 
    J_history = [] # normal list

    for i in range(num_iters):
        h = np.dot(X, theta)
        theta = theta - (alpha / m) * np.dot((h - y), X)
        J_history.append(computeCost(X, y, theta))

    return theta, J_history

theta, J_history = gradientDescent(Age_stack, Cost, theta=np.array([0, 0]), alpha=0.0001, num_iters=100)

print(J_history)
print(theta)

# check if J is decreasing over period of iterations
size_J = len(J_history)
iters = np.linspace(0, 10, size_J)
fig = plt.figure()
plt.plot(iters, J_history)
plt.show()

"""
Use plotData function to add linear regression
"""

plotData(Age, Cost)
plt.plot(Age, np.dot(Age_stack, theta), 'b-')
plt.legend(["Training Data", "Linear Regression"])
plt.show()




"""
Check correctness of J value 
"""
# 2-dim basegrid for J
# theta0_vals = np.linspace(-10, 10, 100)
# theta1_vals = np.linspace(-10, 10, 100)

# J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# for i, theta0 in enumerate(theta0_vals): # retrieve index + theta0 value
#     for j, theta1 in enumerate(theta1_vals):
#         J_vals[i,j] = computeCost(Age_stack, Cost, theta=[theta0, theta1])

# J_vals = J_vals.T

# # surface plot
# fig = plt.figure(figsize=(12, 5))
# ax = fig.add_subplot(121, projection='3d')
# ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
# plt.xlabel('theta0')
# plt.ylabel('theta1')
# plt.title('Surface')

# # contour plot
# # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
# ax = plt.subplot(122)
# plt.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
# plt.xlabel('theta0')
# plt.ylabel('theta1')
# plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
# plt.title('Contour, showing minimum')

