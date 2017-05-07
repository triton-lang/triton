import numpy as np
import keras as kr
import tensorflow as tf
import isaac as sc
import struct

from tools import ProgressBar, load

def train(prefix, OpType, X, y, nepochs = 100):
    progress = ProgressBar('Training')
    model_path = '{}/model.hdf5'.format(prefix)
    #Release ISAAC's driver
    sc.driver.release()
    np.random.seed(0)
    with tf.device('/cpu:0'):
        #Limit GPU memory usage
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        kr.backend.set_session(sess)
        #Features transformation
        X = np.log2(X)
        #Model
        model = kr.models.Sequential()

        for L in [128, 64, 32, 16, 8]:
            model.add(kr.layers.Dense(L, input_dim=X.shape[1]))
            model.add(kr.layers.Activation('relu'))
        model.add(kr.layers.Dense(1))
        model.add(kr.layers.Activation('relu'))
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        #Train
        history = model.fit(X, y, validation_split=0.1, batch_size=64, epochs=nepochs, 
                            verbose=0, callbacks = [kr.callbacks.LambdaCallback(on_epoch_end = lambda i, _: progress.update(i, nepochs))])
        model.save(model_path)
        model = kr.models.load_model(model_path)
        return model


def maximize(OpType, device, model, shapes, V):
    #Build features
    X = np.zeros((V.shape[0], OpType.nparams), dtype=np.uint32)
    X[:, :OpType.nshape_params] = shapes
    X[:, OpType.nshape_params:] = V
    X = X[OpType.check_valid(device, X), :]
    #Model predictions
    predictions = model.predict(np.log2(X), batch_size=8192, verbose=0)
    pred_perfs = np.sort(predictions, axis=0)[::-1]
    pred_idxs = np.argsort(predictions, axis=0)[::-1]
    #Evaluate best predicted models
    ctx = sc.driver.default_context()
    stream = sc.driver.default_stream()
    perf, idx = [], []
    for i, (pred_perf, pred_idx) in enumerate(zip(pred_perfs, pred_idxs)):
        params = X[pred_idx,:][0].astype(int)
        #print(params)
        try:
            y = OpType(params).benchmark(ctx, stream)   
        except RuntimeError:
            continue
        #Update
        perf += [y]
        idx += [pred_idx]
        if len(perf)==100:
            break
    #Return the actual best
    fmax = np.max(perf)
    farg_max = X[pred_idxs[np.argmax(perf)],OpType.nshape_params:]
    return fmax, farg_max[0].astype(np.uint32)


def prune(prefix, OpType, device, model):
    progress = ProgressBar('Pruning')
    #Restore progress
    path = '{}/prune.npz'.format(prefix)
    X, Y = load(path, [('X', OpType.nshape_params), ('Y', OpType.nparams - OpType.nshape_params)])
    Y = Y.astype(np.uint32)
    V = OpType.all_valid(device)
    #Update
    i = Y.shape[0]
    S = OpType.bench_shapes(device)
    nsamples = len(S)
    progress.update(i, nsamples)
    for x in S:
        perf, y = maximize(OpType, device, model, x, V)
        X = np.vstack((X, x))
        Y = np.vstack((Y, y))
        progress.update(i, nsamples)
        np.savez(path, X = X[:i, :], Y = Y[:i, :])
        i += 1
        if i > nsamples: break
    #Remove duplicates
    Y = np.vstack(set(map(tuple, Y)))
    return  Y
