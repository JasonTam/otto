__author__ = 'jason'

import lasagne


def build_vanilla(input_dim, output_dim,
                  num_hidden_units, batch_size=None,
                  drop_p=0.5):
    
    non_lin_fn = lasagne.nonlinearities.rectify
    # non_lin_fn = lasagne.nonlinearities.LeakyRectify()
    # non_lin_fn = lasagne.nonlinearities.tanh

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, input_dim),
    )

    l_in_drop = lasagne.layers.DropoutLayer(
       l_in,
       p=0.4,
    )


    l_hidden1 = lasagne.layers.DenseLayer(
        l_in_drop,
        num_units=num_hidden_units,
        nonlinearity=non_lin_fn,
    )

    l_hidden1_dropout = lasagne.layers.DropoutLayer(
        l_hidden1,
        p=drop_p,
    )

    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=num_hidden_units,
        nonlinearity=non_lin_fn,
    )

    l_hidden2_dropout = lasagne.layers.DropoutLayer(
        l_hidden2,
        p=drop_p,
    )

    l_hidden3 = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=num_hidden_units,
        nonlinearity=non_lin_fn,
    )

    l_hidden3_dropout = lasagne.layers.DropoutLayer(
        l_hidden3,
        p=drop_p,
    )

    l_out = lasagne.layers.DenseLayer(
        l_hidden3_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return l_out


def maxout(incoming, num_units, ds, **kwargs):
    l1a = lasagne.layers.DenseLayer(incoming, nonlinearity=None,
                                    num_units=num_units * ds, **kwargs)
    l1 = lasagne.layers.FeaturePoolLayer(l1a, ds=ds)
    return l1


def build_maxout(input_dim, output_dim,
                 num_hidden_units, batch_size=None,
                 ds=2, drop_p=0.5):

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, input_dim),
    )
    l_in_drop = lasagne.layers.DropoutLayer(
       l_in,
       p=0.4,
    )
    
    l1 = maxout(l_in_drop, num_units=num_hidden_units, ds=ds)

    l1_drop = lasagne.layers.DropoutLayer(
        l1,
        p=drop_p,
    )

    l2 = maxout(l1_drop, num_units=num_hidden_units, ds=ds)

    l2_drop = lasagne.layers.DropoutLayer(
        l2,
        p=drop_p,
    )
    
    l3 = maxout(l2_drop, num_units=num_hidden_units, ds=ds)

    l3_drop = lasagne.layers.DropoutLayer(
        l3,
        p=drop_p,
    )
    l_out = lasagne.layers.DenseLayer(
        l3_drop,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return l_out


