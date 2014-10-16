from sklearn import tree
from sklearn import ensemble
from numpy import array, bincount, mean, std, max, argmax, min, argmin, median


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
    
def train_model(X, Y, profiles, metric):
    print("Building the model...")

    Xmean = mean(X)
    Xstd = std(X)
    X = (X - Xmean)/Xstd

    Y = Y[:, :]
    Ymax = max(Y)
    Y = Y/Ymax

    ref = argmax(bincount(argmin(Y, axis=1))) #most common profile
    cut = int(0.800*X.shape[0]+1)

    #Train the model
    clf = ensemble.RandomForestRegressor(10, max_depth=10).fit(X[:cut,:], Y[:cut,:])

    t = argmin(clf.predict(X[cut:,:]), axis = 1)
    s = array([y[ref]/y[k] for y,k in zip(Y[cut:,:], t)])
    tt = argmin(Y[cut:,:], axis = 1)
    ss = array([y[ref]/y[k] for y,k in zip(Y[cut:,:], tt)])
    print("Testing speedup : mean = %.3f, median = %.3f, min = %.3f,  max %.3f"%(gmean(s), median(s), min(s), max(s)))
    print("Optimal speedup : mean = %.3f, median = %.3f, min = %.3f,  max %.3f"%(gmean(ss), median(ss), min(ss), max(ss)))
