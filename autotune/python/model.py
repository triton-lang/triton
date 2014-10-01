from sklearn import *;
from sklearn import ensemble;
import numpy as np
import scipy as sp
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure 			 import LinearLayer, TanhLayer, SigmoidLayer, SoftmaxLayer, FeedForwardNetwork, BiasUnit
from pybrain.tools.neuralnets import NNregression, Trainer

def train_model(X, Y, profiles):
    #Preprocessing
    Xmean = np.mean(X, axis=0)
    Xstd = np.std(X, axis=0)
    X = (X - Xmean)/Xstd
    Ymax = np.max(Y)
    Y = Y/Ymax

    ref = np.argmax(np.bincount(np.argmax(Y, axis=1))) #most common profile
    #Cross-validation data-sets
    cut = int(0.1*X.shape[0]+1)
    XTr = X[0:cut, :]
    YTr = Y[0:cut, :]
    XTe = X[cut:,:]
    YTe = Y[cut:,:]

    #Train the model
    print("Training the model...")
    ds = SupervisedDataSet(X.shape[1], Y.shape[1])
    for idx, x in enumerate(X):
        ds.addSample(x, Y[idx,:])
    clf = buildNetwork(*[X.shape[1], 100, Y.shape[1]], hiddenclass = TanhLayer, outclass = LinearLayer)
    #print fnn;
    #trainer = RPropMinusTrainer( fnn, dataset=ds, verbose=True);
    trainer = BackpropTrainer( clf, dataset=ds, verbose=True, momentum=0.01, weightdecay=0.01, learningrate=0.002, batchlearning=False)
    trainer.trainUntilConvergence(maxEpochs=100)

    #Evaluate the model
    GFlops = np.empty(XTe.shape[0])
    speedups = np.empty(XTe.shape[0])
    optspeedups = np.empty(XTe.shape[0])
    for i,x in enumerate(XTe):
        predictions = clf.activate(x)
        label = np.argmax(predictions)
        # print YTe[i,label], YTe[i,ref], np.max(YTe[i,:])
        speedups[i] = YTe[i,label]/YTe[i,ref]
        optspeedups[i] = np.max(YTe[i,:])/YTe[i,ref]
        GFlops[i] = YTe[i,ref]*Ymax

    np.set_printoptions(precision=2)
    print("-----------------")
    print("Average testing speedup : %f (Optimal : %f)"%(sp.stats.gmean(speedups), sp.stats.gmean(optspeedups)))
    print("Average GFLOP/s : %f (Default %f, Optimal %f)"%(np.mean(np.multiply(GFlops,speedups)), np.mean(GFlops), np.mean(np.multiply(GFlops,optspeedups))))
    print("Minimum speedup is %f wrt %i GFlops"%(np.min(speedups), GFlops[np.argmin(speedups)]))
    print("Maximum speedup is %f wrt %i GFlops"%(np.max(speedups), GFlops[np.argmax(speedups)]))
    print("--------")

    print clf
