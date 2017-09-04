from tools import ProgressBar
import isaac as sc
import numpy as np
import sys
from tools import load

def update_probabilities(data, counts, ranges):
    for params in data:
        for i, x in enumerate(params):
            counts[i][np.where(ranges[i]==x)] += 1
    return [x/np.sum(x) for x in counts]
    

def benchmarks(prefix, OpType, device, nsamples):
    progress = ProgressBar('Benchmarks')
    
    path = '{}/data.npz'.format(prefix)
    step = 200
    
    X, Y = load(path, [('X', OpType.nparams), ( 'Y', 1)])
    ctx = sc.driver.Context(device)
    stream = sc.driver.Stream(ctx)

    bufX, bufY = np.empty((step, X.shape[1])), np.empty((step, Y.shape[1]))
    
    #Generate data
    nvalid = X.shape[0]
    progress.update(min(nsamples,nvalid), nsamples)
    while nvalid < nsamples:
        P = OpType.generate_valid(device)
        for params in P:
            print(params)
            sys.stdout.flush()
            op = OpType(params)
            try:
                y = op.benchmark(ctx, stream)            
            except:
                continue
            #Update
            bufX[nvalid % step, :] = params
            bufY[nvalid % step, :] = y
            #Save
            nvalid += 1
            if nvalid % step == 0:
                X = np.vstack((X, bufX))
                Y = np.vstack((Y, bufY))
                np.savez(path, X=X, Y=Y)
            progress.update(nvalid, nsamples)
            if nvalid > nsamples: break
    return X, Y
