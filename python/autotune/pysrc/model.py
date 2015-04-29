from sklearn import tree
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
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

def train_model(X, Y, profiles, perf, metric):
    p = np.random.permutation(X.shape[0])
    X = X[p,:]
    Y = Y[p,:]   
    Y = np.array([perf(xx, yy) for xx, yy in zip(X, Y)])
    Y[np.isinf(Y)] = 0 
    #Train the model
    cut = int(0.9*X.shape[0])

    XTr, YTr = X[:cut,:], Y[:cut,:]
    XCv, YCv = X[cut:,:], Y[cut:,:]

    
    nrmses = {}
    for N in range(1,20):
        for depth in range(1,20):
            clf = ensemble.RandomForestRegressor(N, max_depth=depth).fit(XTr, YTr)
            t = np.argmax(clf.predict(XCv), axis = 1)
            y = np.array([YCv[i,t[i]] for i in range(t.size)])
            ground = np.max(YCv[:,:], axis=1)
            nrmses[clf] = nrmse(ground, y)
    clf = min(nrmses, key=nrmses.get)
    print 'The optimal classifer has NRMSE = %.2g (%d estimators and the max depth is %d'%(nrmses[clf], clf.n_estimators, clf.max_depth)
    return clf
