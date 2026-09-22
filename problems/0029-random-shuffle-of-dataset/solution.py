import numpy as np

def shuffle_data(X, y, seed=None):
	np.random.seed(seed)
	indices=np.arange(len(X))
	np.random.shuffle(indices)
	return X[indices],y[indices]
 