"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import pickle

# Neural network estimator imports
import nolearn
import lasagne
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer, prelu
from lasagne.init import GlorotUniform
from lasagne.nonlinearities import softmax, rectify, sigmoid, tanh
from lasagne.updates import nesterov_momentum, rmsprop, adagrad, sgd, adadelta, momentum, adam, adamax
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

from matplotlib import pyplot as plt

import my_io
import dataset
import report
import my_time_utils


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

NEURALNET_NAME = 'best-gif'


def build_nn():
    num_features = 8
    num_classes = 6

    layers = [  # 5 layers: 3 hidden layers
              ('input', InputLayer),
              ('dense0', DenseLayer),
              ('dropout0', DropoutLayer),
              ('dense1', DenseLayer),
              ('dropout1', DropoutLayer),
              ('dense2', DenseLayer),
              ('dropout2', DropoutLayer),
              ('output', DenseLayer)]
    # layer parameters:
    net = NeuralNet(layers=layers,
                    # Input
                    input_shape=(None, num_features),
                    # Dense0
                    dense0_nonlinearity=rectify,
                    dense0_num_units=1200,
                    dropout0_p=0.4,
                    # Dense1
                    dense1_nonlinearity=rectify,
                    dense1_num_units=1200,
                    dropout1_p=0.4,
                    # Dense2
                    dense2_num_units=1200,
                    dense2_nonlinearity=rectify,
                    dropout2_p=0.4,
                    # Output
                    output_num_units=num_classes,
                    output_nonlinearity=softmax,

                    update= nesterov_momentum,
                    update_learning_rate=0.01,
                    update_momentum=0.9,
                    verbose=1,
                    train_split=nolearn.lasagne.TrainSplit(eval_size=0.1),
                    on_epoch_finished=[report.PlotLossesAccuracy(figsize=(16, 12), dpi=200)],
                    max_epochs=500)

    # for item in net.get_params():
    #     print(item)
    return net


def grid_search_fit(nn, X, y, X_val, y_val):
    print('Begin training...')
    tic = my_time_utils.begin()

    parameters = {
                    'update_learning_rate' :[0.01, 0.001],
                    'dropout0_p' : [0.3, 0.4, 0.5],
                    'dropout1_p' : [0.3, 0.4, 0.5],
                    'dropout2_p' : [0.3, 0.4, 0.5]
                    # 'update'    :   [rmsprop, adagrad, sgd, adadelta, momentum, adam, adamax]
                    # 'dense0_nonlinearity' : [tanh, sigmoid, rectify],
                    # 'dense1_nonlinearity' : [tanh, sigmoid, rectify],
                    # 'dense2_nonlinearity' : [tanh, sigmoid, rectify]

    }
    gs = GridSearchCV(nn, parameters, verbose=10)

    gs.fit(X, y)

    print('Grid scores...')
    for item in gs.grid_scores_:
        print(item)

    report.grid_search_scores_report(gs.grid_scores_)

    return gs


def load_model_and_predict(X_val, y_val):
    cls = load_model('./model.pkl')
    report.clasification_report(y_val, cls.predict(X_val))


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
def main():

    # Load the dataset
    print("Loading data...")
    X, y, X_val, y_val = dataset.balanced_validationset()

    print('Building the network')


    nn = build_nn()


    # ''' GRID SEARCH MODEL '''
    # nn = grid_search_fit(nn, X, y, X_val, y_val)


    # ''' SIMPLE MODEL'''
    # print('Begin training...')
    # tic = my_time_utils.begin()

    nn.fit(X, y)

    # print('Finished fitting in ' + str(my_time_utils.elapsed_time(tic)) + 'seconds\n')


    # ''' LOAD MODEL '''
    # # nn = load_model_and_predict(X_val, y_val)


    ''' COMMON '''
    print('Scoring...')
    print(nn.score(X_val, y_val))

    report.score_report([str(nn.score(X_val, y_val))])
    report.clasification_report(y_val, nn.predict(X_val))
    report.plot_confusion_matrix(y_val, nn.predict(X_val))
    report.save_model(nn)


    ## Garbage for grid search
    #    param_grid = {
    #            # 'dense0_num_units' : [10, 200, 1000],
    #            # 'dense1_num_units' : [10, 200, 1000],
    #            # 'dense2_num_units' : [10, 200, 1000],
    #            # 'update_learning_rate' :[0.01, 0.001, 0.0001, 0.00001],
    #            # 'momentum' : [0.9, 0.1, 0.001],
    #            'dense0_nonlinearity' : [tanh, sigmoid, rectify],
    #            'dense1_nonlinearity' : [tanh, sigmoid, rectify],
    #            'dense2_nonlinearity' : [tanh, sigmoid, rectify]
    #           }
    #    nn_grid = GridSearchCV(nn, param_grid=param_grid, cv=None, verbose=1)


if __name__ == '__main__':
    # main()
    # X, y, X_val, y_val = dataset.balanced_validationset()
    # nn = my_io.load_model("MyoGesture/NN-Outputs/PerfilMedio-AKA-BEST/perfil_medio.pkl")
    # print(nn.score(X_val, y_val))
    main()
