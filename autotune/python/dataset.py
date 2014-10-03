import os
import sys
import re
import random
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from pyviennacl.atidlas import FetchingPolicy

def decode(y):
    fetch = [FetchingPolicy.FETCH_FROM_LOCAL, FetchingPolicy.FETCH_FROM_GLOBAL_CONTIGUOUS, FetchingPolicy.FETCH_FROM_GLOBAL_STRIDED]
    y[7] = fetch[y[7]]
    y[8] = fetch[y[8]]
    return y

def resample(X, tbincount, densities, step):
    Xtuples = [tuple(x) for x in X]
    r = random.random()
    while(True):
        if(len(tbincount)==0 or len(densities)==0 or r<=1.0/len(densities)):
            x = np.array([step*random.randint(1,40), step*random.randint(1,40), step*random.randint(1,40)])
        else:
            probs = [1.0/x if x>0 else 0 for x in tbincount]
            distr = np.random.choice(range(tbincount.size), p = probs/np.sum(probs))
            x = densities[distr].sample()[0]
            x = np.maximum(np.ones(x.shape),(x - step/2).astype(int)/step + 1)*step
        if tuple(x) not in Xtuples:
            break
    return x.astype(int)

def generate_dataset(TemplateType, execution_handler):
    I = 50
    step = 64
    path = "./data"

    # print "Getting some good profiles..."
    # X = np.empty((I, 3))
    # t = np.empty(I)
    # profiles = []
    # for i in range(I):
    #     x = resample(X, [], [], step)
    #     y = execution_handler(x)
    #     if y not in profiles:
    #         profiles.append(y)
    #     idx = profiles.index(y)
    #     X[i,:] = x
    #     t[i] = idx
    # densities = [KernelDensity(kernel='gaussian', bandwidth=2*step).fit(X[t==i,:]) for i in range(int(max(t))+1)];
    #
    # print "Generating the dataset..."
    # N = 10000
    # Y = np.empty((N, len(profiles)))
    # X = np.empty((N,3))
    # t = []
    #
    # for i in range(N):
    #     x = resample(X, [], [], step)
    #     for j,y in enumerate(profiles):
    #         T = execution_handler(x, os.devnull, decode(map(int, y)))
    #         Y[i,j] = 2*1e-9*x[0]*x[1]*x[2]/T
    #     idx = np.argmax(Y[i,:])
    #     X[i,:] = x
    #     t = np.argmax(Y[:i+1,], axis=1)
    #     densities[idx].fit(X[t==idx,:])
    #     if i%10==0:
    #         sys.stdout.write('%d data points generated\r'%i)
    #         sys.stdout.flush()
    #
    # np.savetxt(os.path.join(path,"profiles.csv"), profiles)
    # np.savetxt(os.path.join(path,"X.csv"), X)
    # np.savetxt(os.path.join(path,"Y.csv"), Y)

    profiles = np.loadtxt(os.path.join(path,"profiles.csv"))
    X = np.loadtxt(os.path.join(path,"X.csv"))
    Y = np.loadtxt(os.path.join(path,"Y.csv"))

    return X, Y, profiles
