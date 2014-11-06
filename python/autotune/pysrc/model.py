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

def train_model(X, Y, profiles, metric):
    p = np.random.permutation(X.shape[0])
    X = X[p,:]
    Y = Y[p,:]
    #Normalize
    Ymax = np.max(Y)
    Y = Y/Ymax
    #Train the model
    cut = int(0.95*X.shape[0])

    XTr, YTr = X[:cut,:], Y[:cut,:]
    XCv, YCv = X[cut:,:], Y[cut:,:]

    nrmses = {}
    for N in range(1,10):
        for depth in range(1,5):
            clf = ensemble.RandomForestRegressor(N, max_depth=depth).fit(XTr, YTr)
            t = np.argmin(clf.predict(XCv), axis = 1)
            y = np.array([YCv[i,t[i]] for i in range(t.size)])
            nrmses[clf] = nrmse(np.min(YCv[:,:], axis=1), y)
    clf = min(nrmses, key=nrmses.get)

    t = np.argmin(clf.predict(XCv), axis = 1)
    s = np.array([y[0]/y[k] for y,k in zip(YCv, t)])
    tt = np.argmin(YCv, axis = 1)
    ss = np.array([y[0]/y[k] for y,k in zip(YCv, tt)])

    p5 = lambda a: np.percentile(a, 5)
    p25 = lambda a: np.percentile(a, 25)
    p50 = lambda a: np.percentile(a, 50)
    p75 = lambda a: np.percentile(a, 75)
    p95 = lambda a: np.percentile(a, 95)

    print("Percentile     :\t 5 \t 25 \t 50 \t 75 \t 95")
    print("Testing speedup:\t %.2f\t %.2f\t %.2f\t %.2f\t %.3f"%(p5(s), p25(s), p50(s), p75(s), p95(s)))
    print("Optimal speedup:\t %.2f\t %.2f\t %.2f\t %.2f\t %.3f"%(p5(ss), p25(ss), p50(ss), p75(ss), p95(ss)))

    print clf
    return clf
