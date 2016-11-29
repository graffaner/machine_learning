import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_notebook, output_file, show
from os import getcwd
from scipy.optimize import minimize
import plotly.plotly as py
import plotly.graph_objs as go


def mapFeature(X1, X2):
    # create feature matrix containing polynomials of two variables
    # check X1 and X2 shape
    if X1.shape != X2.shape: 
        print('X1 or X2 not in the right shape')

    if X1.shape[1]!=1:
        print('not a vector shape')

    m = X1.shape[0]
    
    degree = 6 # the highest power of the polynomials in the matrix
    X = np.ones_like(X1)
    for i in range(1, degree+1):
        for j in range(0, i+1):
            X = np.append(X, X1**(i-j) * X2**j, axis=1)
            # print(X[0,:])

    # print(X.shape)
    
    return X



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



# costFuncReg: the cost function with regularization
# the cost function will be called in the optimization function
# so the return value of the cost function needs to be scalar
def costFuncReg(theta, X, Y, l):
    # X: features with the first column set as 1 (x0=1)
    # Y: response
    [m, n] = X.shape
    # m: number of training data
    # n: number of features including x0

    theta = theta.reshape(-1, 1) # column vector shape

    h = sigmoid(theta, X)
    # regularized cost function
    # l: regularization parameter
    # shouldn't regularize theta[0]
    J = 1/m * (-Y.T.dot(np.log(h)) - (1 - Y.T).dot(np.log(1 - h))) + l/2/m * theta[1:,:].T.dot(theta[1:,:]) # m by 1 column vector
    J = J.flatten()

    return J[0]



# grad_J: the gradient of the cost function
# the gradient function will be called in the optimization function
# so the return value of the gradient function needs to be 1 dimensional array
def gradFuncReg(theta, X, Y, l):
    # X: features with the first column set as 1 (x0=1)
    # Y: response
    [m, n] = X.shape
    # m: number of training data
    # n: number of features including x0

    theta = theta.reshape(1, -1) # row vector shape

    h = sigmoid(theta, X)
    L = l * np.ones_like(theta)
    L[0] = 0
    dJ = 1/m * ((h - Y).T.dot(X)) + L/m * theta # 1 by n matrix
    dJ = dJ.flatten() # change to 1-D array in order for optimization to work

    return dJ




def main():
    path = getcwd() + '/'
    data2 = pd.read_csv(path+'ex2data2.txt', header=None, index_col=None)
    data2.columns = ['test1', 'test2', 'pass']

    X1 = data2['test1'].values.reshape(-1,1)
    X2 = data2['test2'].values.reshape(-1,1)
    X = mapFeature(X1, X2)
    Y = data2['pass'].values.reshape(-1,1)
    l = 0.1
    
    theta0 = np.zeros((X.shape[1],1))
    cost = costFuncReg(theta0, X, Y, l)
    print(cost)

    # using BFGS optimization function
    res = minimize(costFuncReg, theta0, args=(X, Y, l), method='BFGS', jac=gradFuncReg, options={'disp': True})
    theta_opt = res.x # optimized theta as parameters

    # calculate decision boundary
    h = 100 # number of points per axis for the grid
    X1_mesh, X2_mesh = np.meshgrid(np.linspace(X1.min()-0.5, X1.max()+0.5, h), np.linspace(X2.min()-0.5, X2.max()+0.5, h))
    X_test = mapFeature(X1_mesh.flatten().reshape(-1,1), X2_mesh.flatten().reshape(-1,1))
    Y_test = pred(theta_opt, X_test)
    mesh_test = pd.DataFrame(data={'test1': X1_mesh.flatten(), 'test2': X2_mesh.flatten(), 'pass': Y_test.flatten()})

    # plotting
    # output to static HTML file
    output_file("ex2_data2_reg.html")
    p = figure(
        tools='pan, reset, box_zoom, save',
        x_axis_label='exam1',
        y_axis_label='exam2'
    )

    p.circle(
        mesh_test[mesh_test['pass']==0]['test1'], 
        mesh_test[mesh_test['pass']==0]['test2'],
        size = 3,
        fill_color = 'aqua',
        fill_alpha = 0.3,
        line_color = None,
        legend='boundary: not pass'
    )

    p.circle(
        mesh_test[mesh_test['pass']==1]['test1'], 
        mesh_test[mesh_test['pass']==1]['test2'],
        size = 3,
        fill_color = 'orange',
        fill_alpha = 0.3,
        line_color = None,
        legend='boundary: pass'
    )

    p.circle(
        data2[data2['pass']==0]['test1'],
        data2[data2['pass']==0]['test2'], size=5,
        legend='training: not pass'
    )

    p.cross(
        data2[data2['pass']==1]['test1'],
        data2[data2['pass']==1]['test2'], color='red',
        size=10, legend='training: pass'
    )

    show(p)

    # contour plot using plotly package
    # Y_test_percent = sigmoid(theta_opt, X_test)

    # trace_contour = go.Contour(
    #     z=Y_test_percent.flatten(),
    #     x=X1_mesh.flatten(),
    #     y=X2_mesh.flatten(),
    #     # colorscale=seaborn_to_plotly( sns.color_palette(sns.dark_palette("purple")) ),
    #     contours = dict(coloring='lines', start=0.0, size=0.5, end=1.0),
    #     line = dict(width=2),
    #     ncontours = 2,
    #     showscale = False,
    # )

    # trace_scatter_nopass = go.Scatter(
    #     x = data2[data2['pass']==0]['test1'],
    #     y = data2[data2['pass']==0]['test2'],
    #     mode = 'markers',
    #     name = 'training: no pass',
    #     marker = dict(
    #         color = 'red',
    #         symbol='cross'
    #         )
    # )

    # trace_scatter_pass = go.Scatter(
    #     x = data2[data2['pass']==1]['test1'],
    #     y = data2[data2['pass']==1]['test2'],
    #     mode = 'markers',
    #     name = 'training: pass',
    #     marker = dict(
    #         color = 'blue'
    #         )
    # )

    # py_layout = go.Layout(
    #     width=400,
    #     height=300,
    #     autosize=False
    # )

    # fig = go.Figure(data=[trace_contour, trace_scatter_nopass, trace_scatter_pass], layout=py_layout)

    # url = py.plot(fig, filename='classification_regularization_contour')


if __name__ == '__main__':
    main()
