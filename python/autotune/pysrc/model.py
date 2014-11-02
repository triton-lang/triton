from sklearn import tree
from sklearn import ensemble
import numpy as np

def gmean(a, axis=0, dtype=None):
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a,np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))

def nrmse(y_ground, y):
    N = y.size
    rmsd = np.sqrt(np.sum((y_ground - y)**2)/N)
    return rmsd/(np.max(y_ground) - np.min(y_ground))

def train_model(X, Y, profiles, metric):
    #Shuffle
    p = np.random.permutation(X.shape[0])
    X = X[p,:]
    Y = Y[p,:]
    #Normalize
    Ymax = np.max(Y)
    Y = Y/Ymax
    #Train the model
    cut = int(0.9*X.shape[0])
    nrmses = {}
    for depth in range(1,10):
        clf = ensemble.RandomForestRegressor(5, max_depth=4).fit(X[:cut,:], Y[:cut,:])
        t = np.argmin(clf.predict(X[cut:,:]), axis = 1)
        y = np.array([Y[cut+i,t[i]] for i in range(t.size)])
        y_ground = np.min(Y[cut:,:], axis=1)
        # for i in range(t.size):
        #     print X[cut+i,:], y[i], y_ground[i]
        nrmses[clf] = nrmse(y_ground, y)
        print depth, nrmses[clf]

    clf = min(nrmses, key=nrmses.get)

    return clf
