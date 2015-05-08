from __future__ import print_function
import matplotlib.pyplot as plt

import src.io_tools as iot
import src.transformers as tform
import os
import numpy as np
from time import time
import sys
import copy
import pdb

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

# pipe = tform.pipe
# pipe.fit(np.r_[feats_train[train_ind, :], feats_test])

# -------------------------------------------------------------------------------------

import cPickle as pickle
import gzip


def _load_data():
    # data = pickle.load(open(os.path.join(DATA_DIR, 'transformed_data.p'), 'rb'))
    data = pickle.load(gzip.open(os.path.join(DATA_DIR, 'log_data.pgz'), 'rb'))
    return data


def _load_data2():
    X_train = pipe.transform(feats_fold_train)
    y_train = labels_fold_train
    X_valid = pipe.transform(feats_fold_val)
    y_valid = labels_fold_val
    X_test = pipe.transform(feats_test)
    y_test = -1*np.ones(len(feats_test), dtype=int)

    data = ((X_train, y_train),
            (X_valid, y_valid),
            (X_test, y_test))
    return data.copy()


# -------------------------------------------------------------------------------------

import itertools
import pickle
import sys
import numpy as np
import math
import lasagne
import theano
import theano.tensor as T
import time
from lasagne_nets import net_zoo

NUM_EPOCHS = 10000
# BATCH_SIZE = 1024
# BATCH_SIZE = 2048
BATCH_SIZE_TRAIN = 600
BATCH_SIZE_VAL = 2048
BATCH_SIZE_TEST = 400
NUM_HIDDEN_UNITS = 1024
LEARNING_RATE = 0.01
MOMENTUM = 0.9

COTRAIN_START = 1  # Number of epochs to train before cotraining
COTRAIN_PERIOD = 1   # The grace period (#epochs) to train without adding more cotrained samples

def load_data():
    data = _load_data()
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]

    fn_x = lambda x, name=None: theano.shared(lasagne.utils.floatX(x), name=name)
    fn_y = lambda y, name=None: T.cast(theano.shared(y, name=name), 'int32')

    return dict(
        X_train1=fn_x(X_train, 'X_train1'),  # For model 1
        y_train1=fn_y(y_train, 'y_train1'), 
        X_train2=fn_x(X_train, 'X_train2'),  # For model 2
        y_train2=fn_y(y_train, 'y_train2'),
        X_valid=fn_x(X_valid, 'X_valid'),
        y_valid=fn_y(y_valid, 'y_valid'),
        X_test=fn_x(X_test, 'X_test'),
        y_test=fn_y(y_test, 'y_test'),
        num_examples_train1=X_train.shape[0],  # For model 1
        num_examples_train2=X_train.shape[0],  # For model 2
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1],
        output_dim=len(np.unique(y_train)),
    )

def create_iter_functions(
        output_layer,
        model_num,
        X_tensor_type=T.matrix,
        # batch_size=BATCH_SIZE,
        batch_size_train=BATCH_SIZE_TRAIN,
        batch_size_val=BATCH_SIZE_VAL,
        batch_size_test=BATCH_SIZE_TEST,
        learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    mn = str(model_num)

    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    #batch_slice = slice(
    #    batch_index * batch_size, (batch_index + 1) * batch_size)
    batch_slice_train = slice(
        batch_index * batch_size_train, (batch_index + 1) * batch_size_train)
    batch_slice_val = slice(
        batch_index * batch_size_val, (batch_index + 1) * batch_size_val) 
    batch_slice_test = slice(
        batch_index * batch_size_test, (batch_index + 1) * batch_size_test)  
        
        

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
    winner_prob = T.max(softmax_out[winner_ind, :])

    # Parameter updating
    all_params = lasagne.layers.get_all_params(output_layer)
    updates_train = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)
    

    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates_train,
        givens={
            X_batch: dataset['X_train'+mn][batch_slice_train],
            y_batch: dataset['y_train'+mn][batch_slice_train],
        },
    )

    iter_cotrain = theano.function(
        [batch_index], [winner_ind, winner_x, winner_y, winner_prob],
        givens={
            X_batch: dataset['X_test'][batch_slice_test],
        },
    )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice_val],
            y_batch: dataset['y_valid'][batch_slice_val],
        },
    )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice_test],
            y_batch: dataset['y_test'][batch_slice_test],
        },
    )

    return dict(
        train=iter_train,
        cotrain=iter_cotrain,
        valid=iter_valid,
        test=iter_test,
    )


def unique_hist(x):
    y = np.bincount(x)
    ii = np.nonzero(y)[0]
    return zip(ii, y[ii])


def cotrain(iter_funcs, 
            model_num,
            batch_size_train=BATCH_SIZE_TRAIN,
            batch_size_val=BATCH_SIZE_VAL,
            batch_size_test=BATCH_SIZE_TEST,
            ):
            # batch_size=BATCH_SIZE):

    mn = str(model_num)

    for epoch in itertools.count(1):
        num_batches_train = dataset['num_examples_train'+mn] // batch_size_train
        num_batches_valid = dataset['num_examples_valid'] // batch_size_val
        num_batches_test = dataset['num_examples_test'] // batch_size_test
        print('>>>>>>>>>>' + str(num_batches_train))

        # REBUILD ITERFUNCS EVERY SINGLE GODAMNED TIME
        ### iter_funcs = create_iter_functions(output_layer, model_num=model_num)

        # TRAIN PHASE
        batch_train_losses = []
        for b in range(num_batches_train):
            print('<<<<' + str(b))
            """
            batch_slice_train = slice(b * batch_size_train, (b + 1) * batch_size_train)
            X_batch = dataset['X_train'+mn].get_value()[batch_slice_train]
            y_batch = dataset['y_train'+mn][batch_slice_train].eval()
            batch_train_loss = iter_funcs['train'](X_batch, y_batch)
            """
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        # COTRAIN PHASE
        win_inds = []
        win_xs = []
        win_ys = []
        win_probs = []
        for b in range(num_batches_test):
            """
            batch_slice_test = slice(b * batch_size_test, (b + 1) * batch_size_test)
            X_batch = dataset['X_test'].get_value()[batch_slice_test]
            y_batch = dataset['y_test'][batch_slice_test].eval()
            win_ind, win_x, win_y, win_prob = iter_funcs['cotrain'](X_batch)
            """
            
            win_ind, win_x, win_y, win_prob = iter_funcs['cotrain'](b)
            
            win_xs.append(win_x)
            win_ys.append(win_y)
            win_inds.append(win_ind)
            win_probs.append(win_prob)
        win_inds = np.array(win_inds)
        win_xs = np.array(win_xs)
        win_ys = np.array(win_ys)
        win_class_hist = unique_hist(win_ys)
        win_probs = np.array(win_probs)
        win_prob_avg = win_probs.mean()


        # VALIDATION PHASE
        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            """
            batch_slice_val = slice(b * batch_size_val, (b + 1) * batch_size_val)
            X_batch = dataset['X_valid'].get_value()[batch_slice_val]
            y_batch = dataset['y_valid'][batch_slice_val].eval()
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](X_batch, y_batch)
            """
            
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
            'win': (win_inds, win_ys, win_probs),
            'win_stats': (win_class_hist, win_prob_avg),
        }


def shuffle_unison(a, b, verbose=False):
    """ Shuffles same-length arrays `a` and `b` in unison"""
    if verbose:
        print('x shape: ' + str(a.shape))
        print(a)
        print('y shape: ' + str(b.shape))
        print(b)
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    return c[:, :a.size//len(a)].reshape(a.shape), c[:, a.size//len(a):].reshape(b.shape)

    #rng_state = np.random.get_state()
    #np.random.shuffle(X_train_new)
    #np.random.set_state(rng_state)
    #np.random.shuffle(y_train_new)

def co_update_dataset(dataset, model_num,
                      win_inds, win_ys, win_probs, prob_thresh,
                      batch_size_test=BATCH_SIZE_TEST):                    
    """
    dataset: contains the data for model 1 and model 2
        (mutable and will be modified in-place)
    model_num: the model number for the SOURCE
    The following are associated with the SOURCE dataset:
        win_inds: index with respect to each batch (each ind < batch_size)
        win_ys: predicted class of winners
        win_probs: probability (confidence) of winners
    prob_thresh: confidence threshold to consider movement of winners
    """
    print('Co-train sample movement:')  
    src_n = str(model_num)
    dst_n = str((model_num%2)+1)

    inds_abs = win_inds + np.arange(len(win_inds)) * batch_size_test
    inds_abs = inds_abs[win_probs > prob_thresh]
    y_win_inds = win_probs > prob_thresh
    if 1:#len(inds_abs):
        # Add winners to destination training set
        X_win = dataset['X_test'].get_value()[inds_abs, :]
        y_win = win_ys[y_win_inds]
        X_train_new_dst = np.concatenate([dataset['X_train'+dst_n].get_value(), X_win])
        y_train_new_dst = np.concatenate([dataset['y_train'+dst_n].owner.inputs[0].get_value(), y_win])
        
        X_train_new_dst, y_train_new_dst = shuffle_unison(X_train_new_dst, y_train_new_dst)
        dataset['X_train'+dst_n].set_value(lasagne.utils.floatX(X_train_new_dst))
        dataset['y_train'+dst_n] = T.cast(theano.shared(y_train_new_dst), 'int32')


        # Remove winners from BOTH testing sets 
        X_test_new = np.delete(dataset['X_test'].get_value(), inds_abs, axis=0)
        y_test_new = np.delete(dataset['y_test'].owner.inputs[0].get_value(), inds_abs)
        dataset['X_test'].set_value(lasagne.utils.floatX(X_test_new))
        dataset['y_test'] = T.cast(theano.shared(y_test_new), 'int32')

        # Updating shapes
        dataset['num_examples_train'+dst_n] = X_train_new_dst.shape[0]
        dataset['num_examples_test'] = X_test_new.shape[0]

        # print('inds: ' + str(inds_abs))
        print('# samples moved: ' + str(len(inds_abs)))
        print('X_train shape: ' + str(X_train_new_dst.shape))
        print('X_test shape: ' + str(X_test_new.shape))
        print('transfered class dist: \n' + str(unique_hist(y_win)))
    else:
        print('Winners not confident enough')

    # return dataset



def disp_stats(epoch_d):
    global now
    print('==========================================')
    print("Epoch {} of {} took {:.3f}s".format(
        epoch_d['number'], num_epochs, time.time() - now))
    now = time.time()
    print("  training loss:\t\t{:.6f}".format(epoch_d['train_loss']))
    print("  validation loss:\t\t{:.6f}".format(epoch_d['valid_loss']))
    print("  validation accuracy:\t\t{:.2f} %".format(
        epoch_d['valid_accuracy'] * 100))

    print('------Winner class dist & avg prob--------')
    print(epoch_d['win_stats'])
    print('==========================================')
    sys.stdout.flush()


def write_log(epoch_d, f_path='./log.txt'):
    with open(f_path, 'a') as f:
        f.write('\t'.join([str(x) for x in [epoch_d['number'], epoch_d['valid_loss'], epoch_d['train_loss']]]) + '\n')
    


# def main(num_epochs=NUM_EPOCHS, verbose=True):
if __name__ == '__main__':
    num_epochs = NUM_EPOCHS
    verbose = True
    log = True

    print("Loading data...")
    # NOTE: `dataset1` used with global scope for fishy business
    dataset = load_data()


    print("Building model and compiling functions...")
    net = net_zoo.build_vanilla(
    #output_layer = net_zoo.build_maxout(
        input_dim=dataset['input_dim'],
        output_dim=dataset['output_dim'],
        num_hidden_units=NUM_HIDDEN_UNITS,
        batch_size=None,
    )
    # iter_funcs = create_iter_functions(dataset1, output_layer)
    iter_funcs1 = create_iter_functions(copy.deepcopy(net), model_num=1)
    iter_funcs2 = create_iter_functions(copy.deepcopy(net), model_num=2)

    print("Starting training...")
    now = time.time()
    try:
        net_iter1 = cotrain(iter_funcs1, model_num=1)
        net_iter2 = cotrain(iter_funcs2, model_num=2)
        for ii in range(num_epochs):
            epoch1 = net_iter1.next()

            # COTRAINING BUSINESS for epoch1
            en1 = epoch1['number']
            if (en1 >= COTRAIN_START) and ((en1-1) % COTRAIN_PERIOD == 0):
                print('Cotrain business after model1')
                # If enough iters, move confident predictions to dataset 2
                (win_inds, win_ys, win_probs) = epoch1['win']
                co_update_dataset(dataset, 1,
                                  win_inds, win_ys, win_probs, 
                                  prob_thresh=0.99)
            

            epoch2 = net_iter2.next()

            # COTRAINING BUSINESS for epoch2
            en2 = epoch2['number']
            if (en2 >= COTRAIN_START) and (((en2-1) + math.ceil(COTRAIN_PERIOD/2.)) % COTRAIN_PERIOD == 0):
                print('Cotrain business after model2')
                # If enough iters, move confident predictions to dataset 1
                (win_inds, win_ys, win_probs) = epoch2['win']
                co_update_dataset(dataset, 2,
                                  win_inds, win_ys, win_probs, 
                                  prob_thresh=0.99)

            if log:
                write_log(epoch1, './1.txt')
                write_log(epoch2, './2.txt')

            print_period = 1
            if verbose:
                if (epoch1['number']-1) % print_period == 0:
                    print('---MODEL1 STATS---')
                    disp_stats(epoch1)
                    print('---MODEL2 STATS---')
                    disp_stats(epoch2)
            else:
                pass
            if epoch1['number'] >= num_epochs:
                break

            if epoch1['number'] >= 100 and epoch1['valid_loss'] > 1:
                break

    except KeyboardInterrupt:
        pass

    # return output_layer









