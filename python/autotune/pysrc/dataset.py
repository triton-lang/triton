import os
import sys
import re
import random
import numpy as np

def sample_profiles(execution_handler, generator):
    print "Sampling profiles..."

    t = np.empty(0)
    profiles = []
    for i, x in enumerate(generator):
        print x
        if i==0:
            X = np.empty((0,len(x)))
        y = execution_handler(x)
        if y not in profiles:
            profiles.append(y)
        idx = profiles.index(y)

        X = np.vstack((X, x))
        t = np.append(t, idx)

    idx = int(t[np.argmax(np.linalg.norm(X, axis=1))])
    profiles = [profiles[idx]] + [x for i,x in enumerate(profiles) if i!=idx]
    return profiles

def sample_dataset(prefix_name, profiles, execution_handler, generator):
    P = len(profiles)
    print "Generating the dataset..."
    Y = np.empty((0, P))
    for i,x in enumerate(generator):
        if i==0:
            X = np.empty((0,len(x)))
        new_y = np.zeros(P)
        for j,y in enumerate(profiles):
            T = execution_handler(x, os.devnull, y)
            new_y[j] = T
        X = np.vstack((X, x))
        Y = np.vstack((Y, new_y))
        if i%10==0:
            sys.stdout.write('%d data points generated\r'%i)
            sys.stdout.flush()

    idx = np.argsort(Y[np.argmax(np.linalg.norm(X, axis=1)),:])
    Y = Y[:, idx]
    profiles = [profiles[i] for i in idx]

    dir = os.path.join("data", prefix_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.savetxt(os.path.join(dir,"X.csv"), X)
    np.savetxt(os.path.join(dir,"Y.csv"), Y)
    np.savetxt(os.path.join(dir,"profiles.csv"), profiles)

    return X, Y, profiles
