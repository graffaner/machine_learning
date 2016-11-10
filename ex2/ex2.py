# coding: utf-8

import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_notebook, show
from os import getcwd
from scipy.optimize import minimize


# only for Jupyter notebook to show inline plot
# output_notebook()

"""
In all the functions, theta needs to be the first variable, and then X and then Y.  So in the optimization function used in the gradient descent, theta is the variable
to be optimized, and X and Y are just parameters passed to all the functions since
X and Y don't change.
Make sure (X, Y) parameters are passed with the right sequence into the optimization function
"""

def checkData(theta, X, Y):
    [m, n] = X.shape
    # m: number of training data
    # n: number of features including x0

    # check theta.shape
    if theta.ndim==2:
        if theta.shape[0]==1:
            if theta.shape[1]!=n:
                return False
        elif theta.shape[0]==n:
            if theta.shape[1]!=1:
                return False
        else:
            return False
    elif theta.ndim==1:
        if theta.shape[0]!=n:
            return False
    else:
        return False

    # check Y.shape
    if Y.shape[0]!=m or Y.shape[1]!=1:
        return False

    return True


# sigmoid function
# h_theta(x)=g(X*theta.T)
def sigmoid(theta, X):
    # X: features with the first column set as 1 (x0=1)

    theta = theta.reshape(-1, 1) # column vector shape
    z = X.dot(theta) # m by 1 column vector

    return 1/(1+np.exp(-z))


# function to predict the classification based on sigmoid function
# if sigmoid > 0.5, =1
# if sigmoid < 0.5, =0
def pred(theta, X):
    Y = sigmoid(theta, X)

    for i in range(0, Y.shape[0]):
        Y[i][0] = 1.0 if Y[i][0] >= 0.5 else 0.0

    return Y


# cost_J: the cost function
# the cost function will be called in the optimization function
# so the return value of the cost function needs to be scalar
def cost_J(theta, X, Y):
    # X: features with the first column set as 1 (x0=1)
    # Y: response
    [m, n] = X.shape
    # m: number of training data
    # n: number of features including x0

    h = sigmoid(theta, X)
    J = 1/m * (-Y.T.dot(np.log(h)) - (1 - Y.T).dot(np.log(1 - h))) # m by 1 column vector
    J = J.flatten()

    return J[0]


# grad_J: the gradient of the cost function
# the gradient function will be called in the optimization function
# so the return value of the gradient function needs to be 1 dimensional array
def grad_J(theta, X, Y):
    # X: features with the first column set as 1 (x0=1)
    # Y: response
    [m, n] = X.shape
    # m: number of training data
    # n: number of features including x0

    h = sigmoid(theta, X)
    dJ = 1/m * ((h - Y).T.dot(X))
    dJ = dJ.flatten()

    return dJ


def main():
    # file path
    # data file and the python script need to be in the same folder
    path = getcwd() + '/'
    data1 = pd.read_csv(path+'ex2data1.txt', header=None, index_col=None)
    data1.columns = ['exam1', 'exam2', 'admission']

    # the row of X represents different training data of each feature
    # the column of X represents different features including x0=1
    X_1 = data1[['exam1', 'exam2']].values
    X_1 = np.insert(X_1, 0, 1, axis=1)
    # the row of Y represents response in training data
    Y_1 = data1['admission'].values.reshape(-1, 1)

    # initialize theta0
    theta0_1 = np.zeros(X_1.shape[1])

    # check data shape
    if not checkData(theta0_1, X_1, Y_1):
        print('check the shape of data')

    # grad_J(theta0_1, X_1, Y_1)

    # using BFGS optimization function
    res = minimize(cost_J, theta0_1, args=(X_1, Y_1), method='BFGS', jac=grad_J, options={'disp': True})
    theta_opt_1 = res.x # optimized theta as parameters

    # build decision boundary
    exam1_bound = np.linspace(30, 100, num=50)
    exam2_bound = (theta_opt_1[0]+theta_opt_1[1]*exam1_bound)/(-theta_opt_1[2])

    # plotting
    p = figure(
        tools='pan, reset, box_zoom, save',
        x_axis_label='exam1',
        y_axis_label='exam2'
    )

    p.circle(
        data1[data1['admission']==0]['exam1'],
        data1[data1['admission']==0]['exam2'], size=5,
        legend='not admitted'
    )

    p.cross(
        data1[data1['admission']==1]['exam1'],
        data1[data1['admission']==1]['exam2'], color='red',
        size=10, legend='admitted'
    )

    p.line(exam1_bound, exam2_bound, legend='boundary', color='green')

    show(p)


    X_test_1 = np.array([1, 45, 85]).reshape(1, -1)
    Y_test_1 = sigmoid(theta_opt_1, X_test_1).flatten()[0]
    print('prediction of exam1=45, exam2=85: %0.3f' % Y_test_1)

    Y_pred_1 = pred(theta_opt_1, X_1)
    accuracy = (Y_pred_1 == Y_1).mean()
    print('prediction accuracy of the training set: ' '{:.1%}'.format(accuracy))


if __name__ == '__main__':
    main()
