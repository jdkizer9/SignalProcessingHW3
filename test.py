import random
import numpy as np

print (random.random())
print (np.identity(2)*89)

accel_variance = 1
pooo = np.array([m.T for m in np.matrix(np.random.multivariate_normal([0,0], np.identity(2)*accel_variance, 10))])

print (pooo)

xhat = np.empty(10, dtype=np.dtype(object))
print (xhat)