from sklearn import *;
from sklearn import ensemble;
import numpy as np
import scipy as sp

def train_model(X, Y, profiles):
    #Preprocessing
    scaler = preprocessing.StandardScaler().fit(X);
    X = scaler.transform(X);
    ref = np.argmax(np.bincount(np.argmax(Y, axis=1))) #most common profile

    #Cross-validation data-sets
    cut = int(0.5*X.shape[0]+1);
    XTr = X[0:cut, :];
    YTr = Y[0:cut, :];
    XTe = X[cut:,:];
    YTe = Y[cut:,:];

    #Train the model
    print("Training the model...");
    clf = linear_model.LinearRegression().fit(XTr,YTr);

    #Evaluate the model
    GFlops = np.empty(XTe.shape[0]);
    speedups = np.empty(XTe.shape[0]);
    optspeedups = np.empty(XTe.shape[0]);
    for i,x in enumerate(XTe):
        predictions = clf.predict(x);
        label = np.argmax(predictions);
        speedups[i] = YTe[i,label]/YTe[i,ref];
        optspeedups[i] = np.max(YTe[i,:])/YTe[i,ref];
        GFlops[i] = YTe[i,ref];

    np.set_printoptions(precision=2);
    print("-----------------");
    print("Average testing speedup : %f (Optimal : %f)"%(sp.stats.gmean(speedups), sp.stats.gmean(optspeedups)));
    print("Average GFLOP/s : %f (Default %f, Optimal %f)"%(np.mean(np.multiply(GFlops,speedups)), np.mean(GFlops), np.mean(np.multiply(GFlops,optspeedups))));
    print("Minimum speedup is %f wrt %i GFlops"%(np.min(speedups), GFlops[np.argmin(speedups)]));
    print("Maximum speedup is %f wrt %i GFlops"%(np.max(speedups), GFlops[np.argmax(speedups)]));
    print("--------");

    print clf
