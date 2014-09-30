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
            x = np.array([step*random.randint(1,40), step*random.randint(1,40), step*random.randint(1,40)]);
        else:
            probs = [1.0/x if x>0 else 0 for x in tbincount]
            distr = np.random.choice(range(tbincount.size), p = probs/np.sum(probs))
            x = densities[distr].sample()[0]
            x = np.maximum(np.ones(x.shape),(x - step/2).astype(int)/step + 1)*step
        if tuple(x) not in Xtuples:
            break
    return x.astype(int)

def generate_dataset(TemplateType, execution_handler):
    I = 0
    step = 64
    max_size = 4000
    path = "./data"

    #Tries to resume
    try:
        X = np.loadtxt(open(os.path.join(path, "X.csv"),"rb"))
        t = np.loadtxt(open(os.path.join(path, "t.csv"),"rb"))
        profiles = np.loadtxt(open(os.path.join(path, "profiles.csv"),"rb")).tolist()
        if not isinstance(profiles[0], list):
            profiles = [profiles]
        N = t.size
        X.resize((N+I, 3), refcheck=False)
        t.resize(N+I, refcheck=False)
        print 'Resuming dataset generation...'
    except:
        X = np.empty((I,I))
        t = np.empty(I)
        profiles = []
        N = 0
        pass


    #Generates new data
    print "Getting some good profiles..."
    densities = [KernelDensity(kernel='gaussian', bandwidth=2*step).fit(X[t==i,:]) for i in range(int(max(t))+1)] if N else [];
    X.resize((N+I, 3), refcheck=False)
    t.resize(N+I, refcheck=False)

    for i in range(I):
        tbincount = np.bincount(t[0:i+1].astype(int))
        x = resample(X, tbincount, densities, step)
        y = execution_handler(x)
        if y not in profiles:
            profiles.append(y)
            densities.append(KernelDensity(kernel='gaussian', bandwidth=2*step))
        idx = profiles.index(y)
        X[N+i,:] = x
        t[N+i] = idx
        densities[idx].fit(X[t[0:N+i+1]==idx,:])
        np.savetxt(os.path.join(path,"X.csv"), X)
        np.savetxt(os.path.join(path,"t.csv"), t)
        np.savetxt(os.path.join(path,"profiles.csv"), profiles)

    print "Generating the dataset..."
    N = 500
    Y = np.empty((N, len(profiles)))
    X = np.empty((N,3))
    t = []
    for i in range(N):
        x = resample(X, np.bincount(t), densities, step)
        for j,y in enumerate(profiles):
            T = execution_handler(x, os.devnull, decode(map(int, y)))
            Y[i,j] = 2*1e-9*x[0]*x[1]*x[2]/T
        idx = np.argmax(Y[i,:])
        X[i,:] = x
        t = np.argmax(Y[:i+1,], axis=1)
        densities[idx].fit(X[t==idx,:])

    np.savetxt(os.path.join(path,"Y.csv"), Y)


    return X, Y, profiles
