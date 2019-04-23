# -*- coding: utf-8 -*-
"""
@author: Erting Pan
"""

from __future__ import print_function
import numpy as np
import scipy.io as sio
import scipy.misc
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

input_dimension = 102
num_classes     = 9
window_size      = 27
num_components  = 4 

def ApplyPCA(X, num_components=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca  = PCA(n_components = num_components, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], num_components))
    return newX, pca
def PadWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
def DenseToOneHot(labels_dense, num_classes=16):
    num_labels     = labels_dense.shape[0]
    index_offset   = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()-1] = 1
    return labels_one_hot

data_mat = sio.loadmat('/data/pan/data/paviac/data/Pavia.mat')
data_in  = data_mat['pavia']
mat_gt   = sio.loadmat('/data/pan/data/paviac/data/Pavia_gt.mat')
label    = mat_gt['pavia_gt']
GT       = label.reshape(np.prod(label.shape[:2]),)

labeled_sets = np.load('/data/pan/data/paviac/data/labeled_index.npy')
valid_sets   = np.load('/data/pan/data/paviac/data/valid_index.npy')
test_sets    = np.load('/data/pan/data/paviac/data/test_index.npy')
all_sets     = np.load('/data/pan/data/paviac/data/all_index.npy')

normdata = np.zeros((data_in.shape[0], data_in.shape[1], data_in.shape[2]), dtype=np.float32)
for dim in range(data_in.shape[2]):
    normdata[:, :, dim] = (data_in[:, :, dim] - np.amin(data_in[:, :, dim])) / \
                          float((np.amax(data_in[:, :, dim]) - np.amin(data_in[:, :, dim])))

data_pca,pca = ApplyPCA(data_in,num_components = num_components)
normpca = np.zeros((data_pca.shape[0], data_pca.shape[1],data_pca.shape[2]), dtype=np.float32)
for dim in range(data_pca.shape[2]):
    normpca[:, :, dim] = (data_pca[:, :, dim] - np.amin(data_pca[:, :, dim])) / \
                          float((np.amax(data_pca[:, :, dim]) - np.amin(data_pca[:, :, dim])))
margin = int((window_size - 1) / 2)
padded_data=PadWithZeros(normpca,margin=margin)

class DataSet(object):
  def __init__(self, images):
    self._num_examples = images.shape[0]
    self._images = images
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    
    hsi_batch_pca = np.zeros((batch_size, window_size, window_size, num_components), dtype=np.float32)
    hsi_batch_patch = np.zeros((batch_size, input_dimension), dtype=np.float32)
    col_pca = data_pca.shape[1]
    col     = data_in.shape[1]
    for q1 in range(batch_size):
      hsi_batch_patch[q1] = normdata[(self._images[start + q1] // col), (self._images[start + q1] % col), :]
      hsi_batch_pca[q1]   = padded_data[(self._images[start + q1] // col_pca):
                                          ((self._images[start + q1] // col_pca) + window_size),
                                        (self._images[start + q1] % col_pca):
                                          ((self._images[start + q1] % col_pca) + window_size), :]    
    block = self._images[start:end]
    hsi_batch_label = GT[block]
    hsi_batch_label = DenseToOneHot(hsi_batch_label, num_classes=num_classes)
    return hsi_batch_patch,hsi_batch_pca,hsi_batch_label,

def ReadDatasets():
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = DataSet(labeled_sets)
    data_sets.valid = DataSet(valid_sets)
    data_sets.test  = DataSet(test_sets)
    data_sets.all   = DataSet(all_sets)
    return data_sets
	
def CalAccuracy(true_label, pred_label, class_num):
    M  = 0
    C  = np.zeros((class_num + 1, class_num + 1))
    c1 = confusion_matrix(true_label, pred_label)
    C[0:class_num, 0:class_num] = c1
    C[0:class_num, class_num]   = np.sum(c1, axis=1)
    C[class_num, 0:class_num]   = np.sum(c1, axis=0)
    N = np.sum(np.sum(c1, axis=1))
    C[class_num, class_num] = N  # all of the pixel number
    OA = np.trace(C[0:class_num, 0:class_num]) / N
    every_class = np.zeros((class_num + 3,))
    for i in range(class_num):
        acc = C[i, i] / C[i, class_num]
        M   = M + C[class_num, i] * C[i, class_num]
        every_class[i] = acc

    kappa = (N * np.trace(C[0:class_num, 0:class_num]) - M) / (N * N - M)
    AA = np.sum(every_class, axis=0) / class_num
    every_class[class_num]     = OA
    every_class[class_num + 1] = AA
    every_class[class_num + 2] = kappa
    return every_class, C

def ColorResult(each_class):
	#colorbar = np.array([[255, 105, 180], [255, 0,   255], [147, 112, 219], [0, 0,   255],
	#					  [25,  25,  112], [100, 149, 237], [0,   191, 255], [0, 255, 0], 
	#				      [128, 0,   128], [85, 107, 47],   [128, 128, 0],   [255, 215, 0], 
	#			          [255, 140, 0],   [112, 128, 144], [128, 0,   0],   [0,   0,   0]])
	colorbar = np.array([[0,   0, 255],  [255, 0,   0], [0, 255, 0], 
                         [255, 255, 0],  [0,   100, 0], [255, 0, 255], 
                         [0,   191, 255],[255, 140, 0], [255, 231, 186]])
	data=ReadDatasets()
	all_sets_index = data.all._images
	image       = np.zeros((3, label.shape[0], label.shape[1]), dtype=np.int64)
	groundtruth = np.zeros((3, label.shape[0], label.shape[1]), dtype=np.int64)
	for i in range(len(all_sets_index)):
		row = all_sets_index[i] // label.shape[1]
		col = all_sets_index[i] %  label.shape[1]
		for k in range(1, 17):
			if label[row, col] == k:
				groundtruth[:, row, col] = colorbar[k-1]
			if each_class[i]   == k:
				image[:, row, col] = colorbar[k-1]
	image = np.transpose(image, (1, 2, 0))
	groundtruth = np.transpose(groundtruth, (1, 2, 0))
	scipy.misc.imsave('/data/pan/data/paviac/merge/merge.jpg', image)
	scipy.misc.imsave('/data/pan/data/paviac/merge/gt.jpg', groundtruth)
	return image

