# libs
from pickletools import optimize
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import optimize

"""
In this exercise we are going to predict the 10 year risk of coronary heart disease CHD 
based on different features
> first two will be based on (age) and (cigsPerDay)
"""

# import data
data_health = pd.read_csv(os.path.join("Dataset", "framingham.csv"), delimiter=",")
X, y  = data_health[["age", "cigsPerDay"]].iloc[:], data_health["TenYearCHD"].iloc[:]

# Visualization

def plotData(X, y):
    pos = y == 1 # boolean structure when pos / neg equal to y when some conditions are met
    neg = y == 0

    fig = plt.figure()
    plt.plot(X["age"].loc[pos], X["cigsPerDay"].loc[pos], "k*", lw=2, ms=10) # lw = linewidth, ms = markersize
    plt.plot(X["age"].loc[neg], X["cigsPerDay"].loc[neg], "ko", mfc='y', ms=8, mec='k', mew=1)

    plt.xlabel("Age")
    plt.ylabel("Cigarettes Per Day")
    plt.legend(["High Risk CHD", "Low Risk CHD"])

"""
Application of Logistic Regression
    ----------------
    0. Creation of X based on polynomial degree + Interception term 
    1. Establish costFunction - Preferably Regularized form (Include Sigmoid Func)
    2. Retrieve Theta for the hypothesis h(theta)
    3. Use Optimized form by using fminc 
    4. Validate + Visualize boundary decision
"""

# Feature Mapping - Step 0 - higher polynomial for better fitting 
def mapFeature(X1, X2, degree=6):
    if X1.ndim > 0: 
        out = [np.ones(X1.shape[0])] # list with nestled numpy array
    else: 
        out = [np.ones(1)]

    for i in range(1, degree + 1): # algo structuring the quadratic feature set
        for j in range(i + 1):
            out.append((X1 ** (i-j)) * (X2 ** j))

    if X1.ndim > 0: # turn list to a numpy array / matrix
        return np.stack(out, axis=1)
    else: 
        return np.array(out)

# Basically a finished output of X to use for Logistic Regression
X_intercept = mapFeature(X["age"], X["cigsPerDay"])

# sigmoid function - hypothesis h outcome
def sigmoidFunc(X, theta):
    z = np.dot(X, theta)
    h = 1 / (1 + np.exp(-z))
    return h 

# Cost Function Reg
def costFunctionReg(theta, X, y, lambda_):
    # setup initial parameters
    m = X.shape[0] # number of training data
    J = 0 # initial value
    grad = np.zeros(theta.shape)

    # fill in the parameters
    h = sigmoidFunc(X, theta)
    calc_cf = -y * np.log(h) - (1-y) * np.log((1-h))
    
    # theta grad separation
    temp = theta
    temp[0] = 0 

    # calculation of gradients
    grad = (1 / m) * np.dot((h-y), X) # for j = 0
    grad = grad + (lambda_ / m) * temp

    # Cost Function including gradient part to avoid overfitting
    J = (1 / m) * np.sum(calc_cf) + (lambda_ / (2*m)) * np.sum(np.square(temp))

    return J, grad

"""
Validating the Cost Function Reg
"""
initial_theta = np.zeros(X_intercept.shape[1]) # X polynomial size
lambda_ = 100
J, grad = costFunctionReg(initial_theta, X_intercept, y, lambda_)

# Using the Optimize Func
options = {'maxiter': 400}
res = optimize.minimize(costFunctionReg,
                        initial_theta,
                        (X_intercept, y, lambda_),
                        jac=True,
                        method='TNC',
                        options=options)
    

# ouput using optimized regularized form
J = res.fun
theta = res.x


# step 4 - Validation of data and algo

# adding a predict function to determine the accuracy of the Theta Output from the optimize.minimize func
def predict(theta, X): # input X_intercept
    m = X.shape[0] # number of training examples
    p = np.zeros(m)

    h = sigmoidFunc(X, theta) # predict outcome with given theta

    for i, value_sig in enumerate(h): 
        if value_sig >= 0.5:
            p[i] = 1
        else:
            p[i] = 0 

    return p 


person_1 = np.array([1, 20, 0]) # input data
X_person_1 = mapFeature(person_1[0], person_1[1]) # this include the intercept term
prob = sigmoidFunc(theta, X_person_1)
print("The probability that person 1 has CHD: {:.4f}".format(*prob))


# compute accuracy on our training data set
p = predict(theta, X_intercept)
print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100)) # low accuracy usually means that the Theta could not accurately predict the outcome


"""
Plot results including Decision boundary
""" 
def plotDecisionBoundary(plotData, X, y, theta):
    """
    Plots the data points X and y into a new figure with the decision boundary defined by theta.
    Plots the data points with * for the positive examples and o for  the negative examples.
    Parameters
    ----------
    plotData : func
        A function reference for plotting the X, y data.
    theta : array_like
        Parameters for logistic regression. A vector of shape (n+1, ).
    X : array_like
        The input dataset. X is assumed to be  a either:
            1) Mx3 matrix, where the first column is an all ones column for the intercept.
            2) MxN, N>3 matrix, where the first column is all ones.
    y : array_like
        Vector of data labels of shape (m, ).
    """
    # make sure theta is a numpy array
    theta = np.array(theta)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.xlim([30, 100])
        plt.ylim([30, 100])
    else:
        # Here is the grid range
        u = np.linspace(0, 70, 70)
        v = np.linspace(0, 70, 70)

        z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i, j] = np.dot(mapFeature(ui, vj), theta)

        z = z.T  # important to transpose z before calling contour

        # Plot z = 0
        plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
        plt.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)

plotDecisionBoundary(plotData, X_intercept, y, theta)
plt.show() 