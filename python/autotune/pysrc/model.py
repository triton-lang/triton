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
    
def train_model(X, Y, profiles, metric):
    Y=Y[:,:]
    profiles=profiles[:]
    Ymax = np.max(Y)
    Y = Y/Ymax

    #Train the model
    cut = int(0.75*X.shape[0])
    clf = ensemble.RandomForestRegressor(10, max_depth=4).fit(X[:cut,:], Y[:cut,:])

    print clf.predict([10000])

    t = np.argmin(clf.predict(X[cut:,:]), axis = 1)
    s = np.array([y[0]/y[k] for y,k in zip(Y[cut:,:], t)])
    tt = np.argmin(Y[cut:,:], axis = 1)
    ss = np.array([y[0]/y[k] for y,k in zip(Y[cut:,:], tt)])

    p5 = lambda a: np.percentile(a, 5)
    p25 = lambda a: np.percentile(a, 25)
    p50 = lambda a: np.percentile(a, 50)
    p75 = lambda a: np.percentile(a, 75)
    p95 = lambda a: np.percentile(a, 95)

    print("Percentile     :\t 5 \t 25 \t 50 \t 75 \t 95")
    print("Testing speedup:\t %.2f\t %.2f\t %.2f\t %.2f\t %.3f"%(p5(s), p25(s), p50(s), p75(s), p95(s)))
    print("Optimal speedup:\t %.2f\t %.2f\t %.2f\t %.2f\t %.3f"%(p5(ss), p25(ss), p50(ss), p75(ss), p95(ss)))

    return clf
