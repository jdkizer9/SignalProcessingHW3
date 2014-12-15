import numpy as np
import simulation
import matplotlib.pyplot as plt
import math




def kalmanFilter(T=1, variance_w=.3, variance_v=1000, epsilon=.00001, initialState=[0., 0., 0., 0.], sensorProbability=1., graph=True):


	trueTrajectoryArray, rawMeasurementsArray, trueTrajectoryMatrix, z_T, z_velocity = simulation.generateDataWithMatrices()

	z = np.array([m.T for m in z_T])
	#A = 4x4
	#B = 4x2
	#C = 2x4
	#P = 4x4
	#Q = 4x4
	#R = 2x2
	#K = 4x2
	#U = 2x1
	#X = 4x1
	#Z = 2x1
		# intial parameters
	# n_iter = 50
	# sz = (n_iter,) # size of array
	# x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
	# z = numpy.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)

	#Constant Matrices
	A = np.matrix([[1., T, 0., 0.], [0., 1., 0., 0.], [0., 0., 1., T], [0., 0., 0., 1.]])
	B = np.matrix([[.5*T**2, 0], [T, 0.], [0., .5*T**2], [0, T]])
	C = np.matrix([[1., 0., 0., 0.], [0., 0., 1., 0.]])

	K = np.empty(len(z), dtype=np.dtype(object))

	Pminus = np.empty(len(z), dtype=np.dtype(object))
	P = np.empty(len(z), dtype=np.dtype(object))
	
	Q = np.identity(4)*variance_w
	R = np.identity(2)*variance_v

	accel_variance = .3
	u = np.array([m.T for m in np.matrix(np.random.multivariate_normal([0,0], np.identity(2)*accel_variance, len(z)))])

	xhat = np.empty(len(z), dtype=np.dtype(object))
	xhatminus = np.empty(len(z), dtype=np.dtype(object))
	
	# intial guesses
	xhat[0] = np.matrix(initialState).T
	P[0] = np.matrix([[epsilon, 0., 0., 0.], [0., epsilon, 0., 0.], [0., 0., epsilon, 0.], [0., 0., 0., epsilon]])

	I = np.identity(4)

	for k in range(1,len(z)):
		# time update
		assert(A.shape == (4,4))
		#print(xhat[k-1].shape)
		assert(xhat[k-1].shape == (4,1))
		assert(B.shape == (4,2))
		assert(u[k-1].shape == (2,1))

		xhatminus[k] = A*xhat[k-1] + B*u[k-1]

		assert(xhatminus[k].shape == (4,1))

		#print(P[k-1].shape)
		assert(P[k-1].shape == (4,4))
		assert(Q.shape == (4,4))		

		Pminus[k] = A*P[k-1]*A.T + Q

		assert(Pminus[k].shape == (4,4))


		# measurement update

		assert(C.shape == (2,4))
		assert(R.shape == (2,2))

		K[k] = Pminus[k]*C.T * (C*Pminus[k]*C.T + R).getI()

		assert(K[k].shape == (4,2))

		#print(z[k].shape)
		assert(z[k].shape == (2,1))

		##implement imperfect sensor here

		xhat[k] = xhatminus[k] + K[k]*(z[k]-C*xhatminus[k])

		assert(xhat[k].shape == (4,1))

		P[k] = (I-K[k]*C)*Pminus[k]

		assert(P[k].shape == (4,4))

	#print(xhat)

	xhat_a = np.zeros((len(xhat),4))
	#print(xhat[0].shape)
	for i in range(len(xhat)):
		#print(xhat[0].shape)
		xhat_a[i] = xhat[i].T.getA()

	def computeEstimationError(truth, estimate):
		# print(truth.shape)
		# print(estimate.shape)
		assert(truth.shape == (4,1))
		assert(estimate.shape == (4,1))
		xTrue = truth[0,0]
		yTrue = truth[2,0]
		xEstimate = estimate[0,0]
		yEstimate = estimate[2,0]

		# print(truth)
		# print(estimate)

		return math.sqrt((xEstimate - xTrue)**2 + (yEstimate - yTrue)**2)

	def computeMeasurementError(truth, measurement):
		assert(truth.shape == (4,1))
		assert(measurement.shape == (2,1))

		xTrue = truth[0,0]
		yTrue = truth[2,0]
		xMeasurement = measurement[0,0]
		yMeasurement = measurement[1,0]

		return math.sqrt((xMeasurement - xTrue)**2 + (yMeasurement - yTrue)**2)

	estimationError = np.array([computeEstimationError(trueTrajectoryMatrix[i].T, xhat[i]) for i in range(len(xhat))])
	measurementError = np.array([computeMeasurementError(trueTrajectoryMatrix[i].T, z[i]) for i in range(len(xhat))])

	if(graph):
		plt.figure()
		plt.plot(trueTrajectoryArray[:,0], trueTrajectoryArray[:,2])
		plt.plot(rawMeasurementsArray[:,0], rawMeasurementsArray[:,1])
		plt.plot(xhat_a[:,0], xhat_a[:,2])


		plt.figure()

		plt.plot(list(range(len(estimationError))), np.zeros(len(estimationError)))
		plt.plot(list(range(len(estimationError))), estimationError)
		plt.plot(list(range(len(measurementError))), measurementError)

		# plt.figure()
		# plt.plot(list(range(len(groundTruthData))), groundTruthData[:,1])
		# plt.plot(list(range(len(groundTruthData))),  noisyMeasurement[:,1])

		# plt.figure()
		# plt.plot(list(range(len(groundTruthData))), groundTruthData[:,3])
		# plt.plot(list(range(len(groundTruthData))),  noisyMeasurement[:,3])

		plt.show()

	return estimationError.sum() / measurementError.sum()

if __name__ == "__main__":
	errorRatio = kalmanFilter()
	print(errorRatio)
	variance_w_list=[.003, .03, .3, 3, 30, 300]
	for variance in variance_w_list:
		errorRatio = kalmanFilter(variance_w=variance, graph=False)
		print('For variance_w {} the error ratio is {}'.format(variance, errorRatio))

	errorRatio = kalmanFilter(variance_w=3)




