import os
import re
import random
import numpy as np
from sklearn.neighbors.kde import KernelDensity;

def generate_dataset(operation, execution_handler):
  I = 5
  step = 64;
  max_size = 4000;

  #Retrieves the existing data
  print "Retrieving data..."
  path = "./data"
  files = os.listdir(path)
  X = np.empty((len(files),3))
  t = np.empty(len(files))
  profiles = []
  nonemptyfiles = []
  for i,fname in enumerate(files):
    if os.path.getsize(os.path.join(path,fname))>0:
      nonemptyfiles.append(fname)
  files = nonemptyfiles
  
  for i,fname in enumerate(files):
    MNK = re.search(r"([0-9]+)-([0-9]+)-([0-9]+).csv", fname)
    fl = open(os.path.join(path,fname),"rb")
    A = np.loadtxt(fl,delimiter=',')
    x = np.array([MNK.group(1), MNK.group(2), MNK.group(3)]).astype(float)
    y = tuple(A[np.argmin(A[:,0]),1:])
    if y not in profiles:
      profiles.append(y)
    idx = profiles.index(y)
    X[i,:] = x
    t[i] = idx

  #Generates new data
  print "Generating new data..."
  kdes = [KernelDensity(kernel='gaussian', bandwidth=2*step).fit(X[t==i,:]) for i in range(int(max(t))+1)] if files else [];
  X.resize((len(files)+I, 3), refcheck=False);
  t.resize(len(files)+I, refcheck=False);
    
  max_square = max_size/step
  for i in range(I):
    n_per_label = np.bincount(t[0:i+1].astype(int));
    Xtuples = [tuple(x) for x in X];
    r = random.random();
    while(True):
      if(len(kdes)==0 or r<=1.0/len(kdes)):
        x = np.array([step*random.randint(1,40), step*random.randint(1,40), step*random.randint(1,40)]);
      else:
        probs = (1.0/n_per_label)
        distr = np.random.choice(range(n_per_label.size), p = probs/np.sum(probs))
        x = kdes[distr].sample()[0]
        x = np.maximum(np.ones(x.shape),(x - step/2).astype(int)/step + 1)*step
      if tuple(x) not in Xtuples:
        break;
    x = x.astype(int)
    x = [2048, 2048, 512]
    fname = os.path.join(path, `x[0]` +"-"+ `x[1]` +"-"+ `x[2]` +".csv")
    #Execute auto-tuning procedure
    execution_handler(x, fname)
    #Load csv into matrix
    fl = open(fname,"rb");
    A = np.loadtxt(fl,delimiter=',');
    #Update the kernel density estimators
    y = tuple(A[np.argmin(A[:,0]),1:]);
    if y not in profiles:
      profiles.append(y);
      kdes.append(KernelDensity(kernel='gaussian', bandwidth=2*step));
    idx = profiles.index(y);
    #Update data
    X[len(files)+i,:] = x;
    t[len(files)+i] = idx;
    #Update density estimator p(M,N,K | t=idx)
    kdes[idx].fit(X[t[0:len(files)+i+1]==idx,:]);


  print "Exporting data...";
  #Shuffle the list of file
  files = os.listdir(path)
  random.shuffle(files)
  X = np.empty((len(files),3))
  Y = np.zeros((len(files), len(profiles)))
  for i,fname in enumerate(files):
    MNK = re.search(r"([0-9]+)-([0-9]+)-([0-9]+).csv", fname)
    X[i,:] = map(float,[MNK.group(k) for k in range(1,4)])
    fl = open(os.path.join(path,fname),"rb");
    A = np.loadtxt(fl,delimiter=',')
    for j,y in enumerate(profiles):
      idx = np.where(np.all(A[:,1:]==y,axis=1))[0]
      if idx.size:
        Y[i,j] = 2*1e-9*X[i,0]*X[i,1]*X[i,2]/A[idx[0],0]
      else:
        sys.exit('Data invalid! Were all the data csv files generated using the same auto-tuner options?')
  np.savetxt(export_path+'X.csv', X)
  np.savetxt(export_path+'Y.csv', Y)
  np.savetxt(export_path+'profiles.csv', profiles)
  open(export_path+'pad.csv', 'w').write(str(pad))
