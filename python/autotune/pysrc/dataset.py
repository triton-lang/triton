import os
import sys
import re
import random
import numpy as np

def resample(X, sampler):
    Xtuples = [tuple(x) for x in X]
    r = random.random()
    while(True):
        x = sampler()
        if tuple(x) not in Xtuples:
            break
    return x.astype(int)

def sample_profiles(execution_handler, nTuning, sampler):
    print "Sampling profiles..."
    nDim = sampler().size
    X = np.empty((nTuning, nDim))
    t = np.empty(nTuning)
    profiles = []
    for i in range(nTuning):
        x = resample(X, sampler)
        y = execution_handler(x)
        if y not in profiles:
            profiles.append(y)
        idx = profiles.index(y)
        X[i,:] = x
        t[i] = idx

    idx = int(t[np.argmax(np.linalg.norm(X, axis=1))])
    profiles = np.array([profiles[idx]] + [x for i,x in enumerate(profiles) if i!=idx])
    return profiles

def sample_dataset(prefix_name, profiles, execution_handler, nDataPoints, sampler):

    print "Generating the dataset..."
    Y = np.empty((nDataPoints, len(profiles)))
    X = np.empty((nDataPoints, len(profiles[0])))
    t = []

    for i in range(nDataPoints):
        x = resample(X, sampler)
        for j,y in enumerate(profiles):
            T = execution_handler(x, os.devnull, y)
            Y[i,j] = T
        idx = np.argmax(Y[i,:])
        X[i,:] = x
        t = np.argmax(Y[:i+1,], axis=1)
        if i%10==0:
            sys.stdout.write('%d data points generated\r'%i)
            sys.stdout.flush()

    idx = np.argsort(Y[np.argmax(X),:])
    Y = Y[:, idx]
    profiles = profiles[idx]

    dir = os.path.join("data", prefix_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.savetxt(os.path.join(dir,"X.csv"), X)
    np.savetxt(os.path.join(dir,"Y.csv"), Y)
    np.savetxt(os.path.join(dir,"profiles.csv"), profiles)
    X = np.loadtxt(os.path.join(dir, "X.csv"),ndmin=2)
    Y = np.loadtxt(os.path.join(dir, "Y.csv"),ndmin=2)
    profiles = np.loadtxt(os.path.join(dir, "profiles.csv"))

    return X, Y
