from __future__ import print_function
import matplotlib.pyplot as plt

import src.io_tools as iot
import src.transformers as tform
import os
import numpy as np
from time import time
import sys

from sklearn.cross_validation import StratifiedKFold

# DATA_DIR = iot.DATA_DIR
DATA_DIR = '/media/raid_arr/data/otto/data/'
DB_OUT_TRAIN = os.path.join(DATA_DIR, 'train_lvl')
DB_OUT_TEST = os.path.join(DATA_DIR, 'test_lvl')
DB_OUT_ALL = os.path.join(DATA_DIR, 'all_lvl')
DB_OUT_ALL0 = os.path.join(DATA_DIR, 'all0_lvl')

# -------------------------------------------------------------------------------------

try:
    ids_train, feats_train, labels_train
except NameError:
    ids_train = np.load(os.path.join(iot.DATA_DIR, 'train_ids.npy'))
    feats_train = np.load(os.path.join(iot.DATA_DIR, 'train_feats.npy')).astype(float)
    labels_train = np.load(os.path.join(iot.DATA_DIR, 'train_labels_enc.npy'))

try:
    ids_test, feats_test
except NameError:
    ids_test = np.load(os.path.join(iot.DATA_DIR, 'test_ids.npy'))
    feats_test = np.load(os.path.join(iot.DATA_DIR, 'test_feats.npy')).astype(float)

skf = StratifiedKFold(labels_train, n_folds=5, shuffle=True)
# All
feats_all = np.r_[feats_train, feats_test]
labels_all = np.r_[labels_train, -1*np.ones(len(ids_test))].astype(int)

# All minus test0
train_ind, val_ind = iter(skf).next()
feats_all0 = np.r_[feats_train[train_ind, :], feats_test]
labels_all0 = np.r_[labels_train[train_ind], -1*np.ones(len(ids_test))].astype(int)


feats_fold_train = feats_train[train_ind, :]
labels_fold_train = labels_train[train_ind]
feats_fold_val = feats_train[val_ind, :]
labels_fold_val = labels_train[val_ind]

# -------------------------------------------------------------------------------------

pipe = tform.pipe
pipe.fit(np.r_[feats_train[train_ind, :], feats_test])

# -------------------------------------------------------------------------------------

X_train = pipe.transform(feats_fold_train)
y_train = labels_fold_train

X_valid = pipe.transform(feats_fold_val)
y_valid = labels_fold_val

X_test = pipe.transform(feats_test)
y_test = -1*np.ones(len(feats_test), dtype=int)


def _load_data():
    # X_train = pipe.transform(feats_fold_train)
    y_train = labels_fold_train
    # X_valid = pipe.transform(feats_fold_val)
    y_valid = labels_fold_val
    # X_test = pipe.transform(feats_test)
    y_test = -1*np.ones(len(feats_test), dtype=int)

    data = ((X_train, y_train),
            (X_valid, y_valid),
            (X_test, y_test))
    return data

# -------------------------------------------------------------------------------------

import itertools
import pickle
import sys
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
from lasagne_nets import net_zoo

NUM_EPOCHS = 5000
BATCH_SIZE_TRAIN = 2048
BATCH_SIZE_TEST = 2048
NUM_HIDDEN_UNITS = 1024
LEARNING_RATE = 0.01
MOMENTUM = 0.9


def load_data():
    data = _load_data()
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]

    return dict(
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        X_test=theano.shared(lasagne.utils.floatX(X_test)),
        y_test=T.cast(theano.shared(y_test), 'int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1],
        output_dim=len(np.unique(y_train)),
    )


def create_iter_functions(dataset, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE_TRAIN,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    batch_slice = slice(
        batch_index * batch_size, (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(
        output_layer,
        loss_function=lasagne.objectives.categorical_crossentropy)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch,
                                   deterministic=True)

    softmax_out = output_layer.get_output(X_batch, deterministic=True)

    # Validation Accuracy
    pred = T.argmax(softmax_out, axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    # Co-training winners
    winner_ind = T.argmax(
        T.max(output_layer.get_output(X_batch, deterministic=True), axis=1),
        axis=0)
    winner_x = X_batch[winner_ind, :]
    winner_y = T.argmax(softmax_out[winner_ind, :])


    # Paramter updating
    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)

    iter_train = theano.function(
        # [batch_index], loss_train,
        [batch_index], [loss_train, (winner_x, winner_y)],
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
        },
    )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
        },
    )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
        },
    )

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
    )


def train(iter_funcs, dataset, batch_size=BATCH_SIZE_TRAIN):
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        # TRAIN PHASE
        batch_train_losses = []
        for b in range(num_batches_train):
            # batch_train_loss = iter_funcs['train'](b)
            batch_train_loss, (win_x, win_y) = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        # VALIDATION PHASE
        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
            'lol': (win_x, win_y),
        }


def main(num_epochs=NUM_EPOCHS, verbose=True):
    print("Loading data...")
    dataset = load_data()

    print("Building model and compiling functions...")
    output_layer = net_zoo.build_vanilla(
        input_dim=dataset['input_dim'],
        output_dim=dataset['output_dim'],
        batch_size=BATCH_SIZE_TRAIN,
        num_hidden_units=NUM_HIDDEN_UNITS,
    )
    iter_funcs = create_iter_functions(dataset, output_layer)

    print("Starting training...")
    now = time.time()
    try:
        for epoch in train(iter_funcs, dataset):
            if verbose:
                if (epoch['number']-1) % 10 == 0:
                    print("Epoch {} of {} took {:.3f}s".format(
                        epoch['number'], num_epochs, time.time() - now))
                    now = time.time()
                    print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
                    print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
                    print("  validation accuracy:\t\t{:.2f} %%".format(
                        epoch['valid_accuracy'] * 100))

                    print('--------------')
                    print(epoch['lol'])
                    sys.stdout.flush()
            else:
                pass
            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass

    return output_layer

























