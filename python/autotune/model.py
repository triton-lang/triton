from sklearn import tree
from sklearn import ensemble

from numpy import array, bincount, mean, std, max, argmax, min, argmin, median
from scipy.stats import gmean


# def random_forest(Xtr, Ytr):
#     clf = ensemble.RandomForestRegressor(10, max_depth=7).fit(Xtr,Ytr)
#
#     def predict_tree(tree, x):
#         tree_ = tree.tree_
#         children_left = tree_.children_left
#         children_right = tree_.children_right
#         threshold = tree_.threshold
#         feature = tree_.feature
#         value = tree_.value
#         idx = 0
#         while children_left[idx]!=-1:
#             if x[0, feature[idx]] <= threshold[idx]:
#                 idx = children_left[idx]
#             else:
#                 idx = children_right[idx]
#         return value[[idx],:,:][:,:,0]
#
#     s = 0
#     for e in clf.estimators_:
#         tree_ = e.tree_
#         children_left = tree_.children_left
#         children_right = tree_.children_right
#         threshold = tree_.threshold
#         feature = tree_.feature
#         value = tree_.value
#         s = s + value.size + feature.size + threshold.size + children_right.size + children_left.size
#     print s*4*1e-3
#     return clf, clf.predict
#
# def train_nn(layer_sizes, XTr, YTr, XTe, YTe):
#     optimizer = HF(open(os.devnull, 'w'), 15)
#     optimizer.doCGBacktracking = True
#     net = FeedforwardNeuralNet(layer_sizes, [Act.Tanh() for i in range(len(layer_sizes)-2)], Act.Linear(), 1e-5)
#
#     nbatch=10
#     bsize = XTr.shape[0]/nbatch
#     data = ((XTr[(i%nbatch)*bsize:(i%nbatch+1)*bsize,:], YTr[(i%nbatch)*bsize:(i%nbatch+1)*bsize,:]) for i in range(nbatch))
#     data = HFDataSource(data, bsize, gradBatchSize = nbatch*bsize, curvatureBatchSize = bsize, lineSearchBatchSize =nbatch*bsize, gradBatchIsTrainingSet=True)
#     iters = optimizer.optimize(HFModel(net), data, 300, otherPrecondDampingTerm=net.L2Cost)
#     bestte = collections.deque([float("inf")]*5, maxlen=5)
#     for i,w in enumerate(iters):
#         Diffte = YTe - net.predictions(XTe).as_numpy_array()
#         Difftr = YTr - net.predictions(XTr).as_numpy_array()
#         Ete = np.sum(Diffte**2)
#         Etr = np.sum(Difftr**2)
#         bestte.append(min(min(bestte),Ete))
#         if min(bestte)==max(bestte):
#             print 'Final test error: ', Ete
#             return net, net.predictions
#         print 'Iteration %d | Test error = %.2f | Train error = %.2f'%(i, Ete, Etr)
#     return net, net.predictions

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
