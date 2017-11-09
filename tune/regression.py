import numpy as np
import keras as kr
import isaac as sc
import struct
from operations import bench_shapes, evaluate, keep_valid, valid_shapes, tuning_ranges
from keras import backend as K
from tools import ProgressBar, load, cartesian_iterator
from keras.layers import Activation, Dense

def train(OpType, X, y, nepochs = 150):
    progress = ProgressBar('Training')
    np.random.seed(0)
    #Features transformation
    X = np.log2(X)
    #Model
    model = kr.models.Sequential()
    for i,L in enumerate([64, 48, 32, 16, 8]):
        model.add(kr.layers.Dense(L, input_dim=X.shape[1]))
        model.add(kr.layers.Activation('relu'))
    model.add(kr.layers.Dense(1))
    model.add(kr.layers.Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #Train
    history = model.fit(X, y, validation_split=0.1, batch_size=32, epochs=nepochs,
                        verbose=1, callbacks = [kr.callbacks.LambdaCallback(on_epoch_end = lambda i, _: progress.update(i, nepochs))])
    return model


def maximize(OpType, model, shapes, V, device, ctx, stream):
    # Build features
    S = np.tile(shapes, (V.shape[0], 1))
    X = np.concatenate((S, V), axis=1)
    X = keep_valid(OpType, device, X)
    # Model predictions
    predictions = model.predict(np.log2(X), batch_size=8192, verbose=0)
    pred_idxs = np.argsort(predictions, axis=0)[::-1]
    # Actual evaluation of the few best configurations
    perf, idx = [], []
    for i, pred_idx in enumerate(pred_idxs):
        params = X[pred_idx,:][0].astype(int)
        try:
            y = evaluate(OpType, ctx, stream, params)
        except RuntimeError:
            continue
        #Update
        perf += [y]
        idx += [pred_idx]
        if len(perf)==100:
            break
    #Return the actual best
    fmax = np.max(perf)
    farg_max = X[pred_idxs[np.argmax(perf)],OpType.Nshapes:]
    return fmax, farg_max[0].astype(np.uint32)


def valid_configurations(OpType, device):
    Nparams = OpType.Nparams
    Nshapes = OpType.Nshapes
    Ntune = OpType.Ntune
    X = np.empty((0, Ntune))
    for s in valid_shapes(OpType):
        for T in cartesian_iterator(tuning_ranges(OpType)):
            S = np.tile(s, (T.shape[0],1))
            Y = np.concatenate((S, T), axis = 1)
            Xnew = keep_valid(OpType, device, Y)[:, Nshapes:]
            X  = np.vstack((X, Xnew))
    return X

def prune(OpType, model, init_cuda):
    progress = ProgressBar('Pruning')
    device, ctx, stream = init_cuda()
    #Restore progress
    X = np.empty((0, OpType.Nshapes))
    Y = np.empty((0, OpType.Nparams - OpType.Nshapes), dtype=np.uint32)
    V = valid_configurations(OpType, device)
    #Update
    i = Y.shape[0]
    S = bench_shapes(OpType, device)
    for i, x in enumerate(S):
        perf, y = maximize(OpType, model, x, V, device, ctx, stream)
        X = np.vstack((X, x))
        Y = np.vstack((Y, y))
        progress.update(i, len(S))
        print(x, perf)
    #Remove duplicates
    Y = np.vstack(set(map(tuple, Y)))
    return  Y
