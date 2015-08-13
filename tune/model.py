from isaac.external.forest import RandomForestRegressor
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
    if len(y_ground) > 1:
        return rmsd/(np.max(y_ground) - np.min(y_ground))
    else:
        return rmsd

def train(X, Y, profiles):      
    X = np.array(X)
    Y = np.array(Y)
    M = X.shape[0]

    p = np.random.permutation(X.shape[0])
    X = X[p,:]
    Y = Y[p,:]   

    #Train the.profile
    cut = int(1.00*M)
    CV = .1
    XTr, YTr = X[:,:], Y[:,:]
    XCv, YCv = X[:max(1,CV*M),:], Y[:max(1,CV*M),:]

    nrmses = {}
    for N in range(1,min(M+1,20)):
        for depth in range(1,min(M+1,20)):
            clf = RandomForestRegressor(N, max_depth=depth).fit(XTr, YTr)
            t = np.argmax(clf.predict(XCv), axis = 1)
            y = np.array([YCv[i,t[i]] for i in range(t.size)])
            ground = np.max(YCv[:,:], axis=1)
            nrmses[clf] = nrmse(ground, y)
            
    clf = min(nrmses, key=nrmses.get)
    return clf, nrmses[clf]
