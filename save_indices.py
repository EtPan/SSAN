 # -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio

def Sampling(groundtruth):              #divide dataset into train and test datasets
    labeled = {}
    test    = {}
    valid   = {}
    all     = {}
    m = max(groundtruth)
    labeled_indices = []
    test_indices    = []
    valid_indices   = []
    all_indices     = []
    for i in range(m+1):
        indices = [j for j, x in enumerate(groundtruth.ravel().tolist()) if x == i]
        if i != 0:
            np.random.shuffle(indices)
            all[i]     = indices
            test[i]    = indices[200:]
            valid[i]   = indices[100:200]
            labeled[i] = indices[:100]
            labeled_indices += labeled[i]
            valid_indices   += valid[i]
            test_indices    += test[i]            
            all_indices     += all[i]

    np.random.shuffle(labeled_indices)
    np.random.shuffle(valid_indices)
    np.random.shuffle(test_indices)
    np.random.shuffle(all_indices)
    return labeled_indices, test_indices, valid_indices,all_indices

mat_gt = sio.loadmat("/data/pan/data/paviac/data/Pavia_gt.mat")
label  = mat_gt['pavia_gt']
GT     = label .reshape(np.prod(label.shape[:2]),)

labeled_indices, test_indices, valid_indices,all_indices= Sampling(GT)

np.save('/data/pan/data/paviac/data/labeled_index.npy', labeled_indices)
np.save('/data/pan/data/paviac/data/valid_index.npy', valid_indices)
np.save('/data/pan/data/paviac/data/test_index.npy', test_indices)
np.save('/data/pan/data/paviac/data/all_index.npy', all_indices)




 


