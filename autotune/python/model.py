from sklearn import *;
from sklearn import ensemble;
import numpy as np
import scipy as sp
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure 			 import LinearLayer, TanhLayer, SigmoidLayer, SoftmaxLayer, FeedForwardNetwork, BiasUnit
from pybrain.tools.neuralnets import NNregression, Trainer

def train_model(X, Y, profiles, metric):
    #Preprocessing
    Xmean = np.mean(X, axis=0)
    Xstd = np.std(X, axis=0)
    X = (X - Xmean)/Xstd

    Ymax = np.max(Y)
    Y = Y/Ymax

    ref = np.argmax(np.bincount(np.argmax(Y, axis=1))) #most common profile
    #Cross-validation data-sets
    cut = int(0.800*X.shape[0]+1)
    XTr = X[0:cut, :]
    YTr = Y[0:cut, :]
    XTe = X[cut:,:]
    YTe = Y[cut:,:]

    #Train the model
    print("Training the model...")
    clf = ensemble.RandomForestRegressor(40).fit(XTr,YTr)

    #Evaluate the model
    GFlops = np.empty(XTe.shape[0])
    speedups = np.empty(XTe.shape[0])
    optspeedups = np.empty(XTe.shape[0])
    for i,x in enumerate(XTe):
        predictions = clf.predict(x)
        label = np.argmax(predictions)
        speedups[i] = YTe[i,label]/YTe[i,ref]
        optspeedups[i] = np.max(YTe[i,:])/YTe[i,ref]
        GFlops[i] = YTe[i,ref]*Ymax

    np.set_printoptions(precision=2)
    print("-----------------")
    print("Average testing speedup : %f (Optimal : %f)"%(sp.stats.gmean(speedups), sp.stats.gmean(optspeedups)))
    print("Average %s: %f (Default %f, Optimal %f)"%(metric, np.mean(np.multiply(GFlops,speedups)), np.mean(GFlops), np.mean(np.multiply(GFlops,optspeedups))))
    print("Minimum speedup is %f wrt %i %s"%(np.min(speedups), GFlops[np.argmin(speedups)], metric))
    print("Maximum speedup is %f wrt %i %s"%(np.max(speedups), GFlops[np.argmax(speedups)], metric))
    print("--------")