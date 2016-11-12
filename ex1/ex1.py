# coding: utf-8

import os
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_notebook, output_file, show

# to plot inline in Jupyter Notebook
# output_notebook()


def h_theta(theta, X):
    # X: features with the first column set as 1 (x0=1)
    theta = theta.reshape(-1, 1) # column vector shape

    if X.ndim!=2:
        print('wrong X dimension, 2D array expected')
        return None
    else:
        # m: number of training data
        # n: number of feature including x0
        (m, n) = X.shape

    h = X.dot(theta)

    return h



def costJ(theta, X, Y):
    # X: features with the first column set as 1 (x0=1)
    # Y: response as a column vector
    theta = theta.reshape(-1, 1) # column vector shape

    if X.ndim!=2:
        print('wrong X dimension, 2D array expected')
        return None
    else:
        # m: number of training data
        # n: number of feature including x0
        (m, n) = X.shape

    Y = Y.reshape(-1, 1)

    h = h_theta(theta, X)
    cost = (h - Y).T.dot(h - Y)
    cost = cost.flatten()

    return cost[0]



def gradientDescent(X, Y, iteration, alpha):
    # gradient descent algorithm
    # X: features with the first column set as 1 (x0=1)
    # Y: response as a column vector
    # iteration: int
    # alpha: float
    # return: theta, a ndarray of floats

    if X.ndim!=2:
        print('wrong X dimension, 2D array expected')
        return None
    else:
        # m: number of training data
        # n: number of feature including x0
        (m, n) = X.shape

    Y = Y.reshape(-1, 1)

    theta = np.zeros(n) # returnn theta
    temp = np.zeros(n) # for updating theta

    J_cost = np.zeros(iteration) # store cost function J(theta)

    for i in range(0, iteration):
        # calculate cost function J
        J_cost[i] = costJ(theta, X, Y)

        # calculate new theta
        h = h_theta(theta, X)
        temp = theta - alpha / m * (h - Y).T.dot(X)

        # update theta simultaneously
        theta=temp

    return (theta, J_cost)



# feature normalization
def normalizeFeature(X):
    # feature normalization
    # X does NOT contain x0=1 column before normalization

    if X.ndim!=2:
        print('wrong X dimension, 2D array expected')
        return None
    else:
        # m: number of training data
        # n: number of feature including x0
        (m, n) = X.shape

    # data after normalization
    norm_data=np.zeros_like(X)

    rescale_factor = np.vstack((X.mean(axis=0), X.std(axis=0)))
    rescale_factor = pd.DataFrame(rescale_factor, index=['mean', 'std'])

    norm_data = (X - rescale_factor.ix['mean'].values) / rescale_factor.ix['std'].values

    return (norm_data, rescale_factor)



# linear regression using norm equation
def normEquation(X, Y):
    # X: features with the first column set as 1 (x0=1)
    # Y: response as a column vector
    X = np.matrix(X)
    Y = np.matrix(Y)
    theta = (X.T * X).I * (X.T) * Y
    theta = theta.A1

    return theta



def main():
    # ex1 data1
    path=os.getcwd()+'/'
    data=pd.read_csv(path+'ex1data1.txt', header=None, index_col=None)
    data.columns=['population', 'profit']

    X_1=data['population'].values.reshape(-1, 1)
    X_1 = np.insert(X_1, 0, 1, axis=1) # add x0=1
    Y_1=data['profit'].values.reshape(-1, 1)

    iteration=1500
    alpha=0.01

    (theta_data1, J_cost_data1) = gradientDescent(X_1, Y_1, iteration, alpha)

    # output to static HTML file
    output_file("ex1_data1_costJ.html")
    # plot J_cost vs. iteration
    p=figure(
        tools='pan, reset, box_zoom, save',
        # y_axis_type='log',
        x_axis_label='Iteration',
        y_axis_label='J(theta)'
    )
    p.circle(range(0, J_cost_data1.size), J_cost_data1)
    show(p)

    # plot raw data and linear regression
    output_file("ex1_data1_regression.html")
    p=figure(
        tools='pan, reset, box_zoom, save',
        x_axis_label='Population of City in 10,000s',
        y_axis_label='Profit in $10,000s'
    )
    pred = h_theta(theta_data1, X_1).flatten()
    p.circle(data['population'], data['profit'])
    p.line(data['population'], pred, color='red')
    show(p)


    # ex1 data2
    path = os.getcwd()+'/'
    housing_data = pd.read_csv(path+'ex1data2.txt', header=None, index_col=None)
    housing_data.columns = ['size', 'bedrooms', 'price']

    X_2 = housing_data[['size', 'bedrooms']].values
    # feature normalization
    (X_2, rescale_factor) = normalizeFeature(X_2)
    X_2 = np.insert(X_2, 0, 1, axis=1) # add x0=1
    Y_2 = housing_data['price'].values.reshape(-1, 1)

    iteration=1500
    alpha=0.01

    (theta_data2, J_cost_data2) = gradientDescent(X_2, Y_2, iteration, alpha)

    # plot J_cost vs. iteration
    output_file("ex1_data2_costJ.html")
    p=figure(
        tools='pan, reset, box_zoom, save',
        # y_axis_type='log',
        x_axis_label='Iteration',
        y_axis_label='J(theta)'
    )
    p.circle(range(0, J_cost_data2.size), J_cost_data2)
    show(p)
    # test data prediction
    test_data2 = np.array([[1650, 3]])
    test_data2 = (test_data2 - rescale_factor.ix['mean'].values) / rescale_factor.ix['std'].values
    test_data2 = np.insert(test_data2, 0, 1, axis=1)

    predict_data2 = h_theta(theta_data2, test_data2)
    predict_data2 = predict_data2.flatten()
    print(predict_data2[0])

    # regression using norm equation
    X_2_norm = housing_data[['size', 'bedrooms']].values
    X_2_norm = np.insert(X_2_norm, 0, 1, axis=1) # add x0=1
    Y_2_norm = housing_data['price'].values.reshape(-1, 1)

    theta_data2_norm = normEquation(X_2_norm, Y_2_norm)

    predict_data2_norm = h_theta(theta_data2_norm, np.array([[1, 1650, 3]]))
    predict_data2_norm = predict_data2_norm.flatten()
    print(predict_data2_norm[0])



if __name__ == '__main__':
    main()
