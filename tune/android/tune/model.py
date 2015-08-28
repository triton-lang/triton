from isaac.external.sklearn.forest import RandomForestRegressor
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
    #Remove unused profiles
    unused = np.where(np.bincount(np.argmax(Y, 1))==0)[0]
    profiles = [x in profiles for ix,x in enumerate(profiles) if ix not in unused]
    Y = np.delete(Y, np.where(np.bincount(np.argmax(Y, 1))==0), axis=1)
    #Shuffle
    p = np.random.permutation(X.shape[0])
    M = X.shape[0]
    X = X[p,:]
    Y = Y[p,:]   

    #Train the.profile
    cut = int(.9*M)
    XTr, YTr = X[:cut,:], Y[:cut,:]
    XCv, YCv = X[cut:,:], Y[cut:,:]

    nrmses = {}
    for N in range(1,min(M+1,10)):
        for depth in range(1,min(M+1,20)):
            clf = RandomForestRegressor(N, max_depth=depth).fit(XTr, YTr)
            t = np.argmax(clf.predict(XCv), axis = 1)
            y = np.array([YCv[i,t[i]] for i in range(t.size)])
            ground = np.max(YCv[:,:], axis=1)
            nrmses[clf] = nrmse(ground, y)
         
    clf = min(nrmses, key=nrmses.get)
    return clf, nrmses[clf]
