__author__ = 'jason'

import lasagne


def build_vanilla(input_dim, output_dim,
                  batch_size, num_hidden_units):

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, input_dim),
    )
    l_hidden1 = lasagne.layers.DenseLayer(
        l_in,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
    )

    l_hidden1_dropout = lasagne.layers.DropoutLayer(
        l_hidden1,
        p=0.5,
    )

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
    )

    l_hidden2_dropout = lasagne.layers.DropoutLayer(
        l_hidden2,
        p=0.5,
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return l_out


def build_maxout(input_dim, output_dim,
                 batch_size, num_hidden_units):

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, input_dim),
    )
    l_hidden1 = lasagne.layers.DenseLayer(
        l_in,
        num_units=num_hidden_units,
        # nonlinearity=lasagne.nonlinearities.rectify,
        nonlinearity=lasagne.nonlinearities.identity,
    )

    lh1m = lasagne.layers.FeaturePoolLayer(
        l_hidden1,
        ds=2,
    )

    l_hidden1_dropout = lasagne.layers.DropoutLayer(
        lh1m,
        p=0.5,
    )

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=num_hidden_units,
        # nonlinearity=lasagne.nonlinearities.rectify,
        nonlinearity=lasagne.nonlinearities.identity,
    )

    lh2m = lasagne.layers.FeaturePoolLayer(
        l_hidden2,
        ds=2,
    )

    l_hidden2_dropout = lasagne.layers.DropoutLayer(
        lh2m,
        p=0.5,
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return l_out