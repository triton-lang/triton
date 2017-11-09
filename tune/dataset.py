from tools import ProgressBar
import isaac as sc
import numpy as np
import sys
from tools import load
from operations import evaluate, keep_valid, num_ops, tuning_ranges, input_ranges
import multiprocessing
from tools import mkdir
from time import time, sleep

def generate_valid(OpType, device):
    ranges = input_ranges(OpType, device) + tuning_ranges(OpType)
    # Random profiles
    X = np.array([np.random.choice(x, size=10000) for x in ranges]).astype(np.uint32).T.copy()
    # Keep valid
    X = keep_valid(OpType, device, X)
    # Prune expensive
    flops = num_ops(OpType, X)
    idx = np.logical_and(flops > 1e7, flops < 1e12)
    return X[idx, :]

def benchmarks_impl(i, OpType, nsamples, init_cuda, data, progress):
    # Initialize CUDA context
    device, ctx, stream = init_cuda()
    # Process-specific seed
    np.random.seed(int(time()) + i)
    # Retrieve saved data
    arch = 'sm_' + '_'.join(map(str, device.compute_capability))
    path = mkdir('save/{}/{}/'.format(OpType.id, arch)) + 'data{}.npz'.format(i)
    X, Y = load(path, [('X', OpType.Nparams), ( 'Y', 1)])
    # Do not update/realloc X, Y at each iteration
    step = 200
    bufX, bufY = np.empty((step, X.shape[1])), np.empty((step, Y.shape[1]))
    #Generate data
    nvalid = X.shape[0]    
    progress[i] = min(nsamples,nvalid)        
    while nvalid < nsamples:
        P = generate_valid(OpType, device)
        for params in P:
            #print(params)
            sys.stdout.flush()
            try:
                y = evaluate(OpType, ctx, stream, params)
            except:
                print('Exception for', params)
                pass
            bufX[nvalid % step, :] = params
            bufY[nvalid % step, :] = y
            # Save
            nvalid += 1
            if nvalid % step == 0:
                X = np.vstack((X, bufX))
                Y = np.vstack((Y, bufY))
                np.savez(path, X=X, Y=Y)
            # Update progress
            progress[i] = min(nsamples,nvalid)
            if nvalid > nsamples:
                break
    data[i] = (X, Y)

def show_progress(progress, nsamples):
    bar = ProgressBar('Benchmarks')
    while True:
        sleep(0.1)
        current = np.sum(progress.values())
        bar.update(current, nsamples)
        if(current > nsamples - 1):
            break
    
def benchmarks(OpType, nsamples, init_cuda, num_workers = 8):
    # Shared
    manager = multiprocessing.Manager()
    data = manager.dict()
    progress = manager.dict()
    # Launch processes
    jobs = []
    per_worker = nsamples/num_workers
    for i in range(num_workers):
        jobs.append(multiprocessing.Process(target=benchmarks_impl, args=(i, OpType, per_worker, init_cuda, data, progress)))
    jobs.append(multiprocessing.Process(target=show_progress, args=(progress,nsamples)))
    # Start
    for job in jobs:
        job.start()
    # Synchronize
    for job in jobs:
        job.join()
    # Concatenate
    X, Y = np.empty((0,OpType.Nparams)), np.empty((0, 1))
    for Xt, Yt in data.values():
        X = np.vstack((X, Xt))
        Y = np.vstack((Y, Yt))
    return X, Y
    
