import os
import sys
import re
import random
import numpy as np
from pyviennacl.atidlas import FetchingPolicy

def resample(X, draw):
    Xtuples = [tuple(x) for x in X]
    r = random.random()
    while(True):
        x = draw()
        if tuple(x) not in Xtuples:
            break
    return x.astype(int)

def generate_dataset(TemplateType, execution_handler, nTuning, nDataPoints, draw):

    print "Getting some good profiles..."
    nDim = draw().size
    X = np.empty((nTuning, nDim))
    t = np.empty(nTuning)
    profiles = []
    for i in range(nTuning):
        x = resample(X, draw)
        y = execution_handler(x)
        if y not in profiles:
            profiles.append(y)
        idx = profiles.index(y)
        X[i,:] = x
        t[i] = idx

    print "Generating the dataset..."
    Y = np.empty((nDataPoints, len(profiles)))
    X = np.empty((nDataPoints, nDim))
    t = []

    for i in range(nDataPoints):
        x = resample(X, draw)
        for j,y in enumerate(profiles):
            T = execution_handler(x, os.devnull, y)
            Y[i,j] = T
        idx = np.argmax(Y[i,:])
        X[i,:] = x
        t = np.argmax(Y[:i+1,], axis=1)
        if i%10==0:
            sys.stdout.write('%d data points generated\r'%i)
            sys.stdout.flush()

    template_name = TemplateType.__name__
    dir = os.path.join("data", template_name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    np.savetxt(os.path.join(dir,"profiles.csv"), profiles)
    np.savetxt(os.path.join(dir,"X.csv"), X)
    np.savetxt(os.path.join(dir,"Y.csv"), Y)

    profiles = np.loadtxt(os.path.join(dir, "profiles.csv"))
    X = np.loadtxt(os.path.join(dir, "X.csv"),ndmin=2)
    Y = np.loadtxt(os.path.join(dir, "Y.csv"),ndmin=2)

    return X, Y, profiles
