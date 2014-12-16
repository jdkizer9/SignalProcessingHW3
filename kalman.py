import numpy as np
import simulation
import matplotlib.pyplot as plt
import math, random	




def kalmanFilter(T=1, variance_w=.3*.3, variance_v=1000*1000, raw_measurement_variance=1000*1000,epsilon=.00001, initialState=[0., 0., 0., 0.], sensorProbability=1., graph=True):


	trueTrajectoryArray, rawMeasurementsArray, trueTrajectoryMatrix, z_T, z_velocity = simulation.generateDataWithMatrices(measurementNoise=([0,0], [[raw_measurement_variance,0],[0,raw_measurement_variance]]))

	z = np.array([m.T for m in z_T])


	my_z = np.empty(len(z), dtype=np.dtype(object))
	#matrix to support prediction
	D = np.matrix([[1., 1., 0., 0.], [0., 0., 1., 1.]])
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
	
	#Q = np.identity(4)*variance_w
	Q = B*(np.identity(2)*variance_w)*B.T

	print(B*(np.identity(2)*variance_w)*B.T)

	R = np.identity(2)*variance_v

	accel_variance = .3
	#u = np.array([m.T for m in np.matrix(np.random.multivariate_normal([0,0], np.identity(2)*accel_variance, len(z)))])
	u = np.array([np.matrix([0., 0.]).T for i in range(len(z))])

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
		#print(K[k])

		assert(K[k].shape == (4,2))

		#print(z[k].shape)
		assert(z[k].shape == (2,1))

		##implement imperfect sensor here
		if random.random() > sensorProbability and k > 1:
			# sensor broke, carry forward old data to make projections
			#xhat[k] = xhatminus[k-1] + K[k-1]*(z[k-1]-C*xhatminus[k-1])

			#create z[k]
			#predict z[k] based on previous location + previous trajectory + noise

			previousStateVector = np.matrix([my_z[k-1][0,0], xhat[k-1][1,0], my_z[k-1][1,0], xhat[k-1][3,0]]).T
			assert(previousStateVector.shape == (4,1))
			assert(D.shape == (2,4))
			estimatedPosition = (D*previousStateVector).T
			assert(estimatedPosition.shape == (1,2))

			#print(estimatedPosition)
			#print(estimatedPosition.getA()[0])


			my_z[k] = np.matrix(np.random.multivariate_normal(estimatedPosition.getA()[0], R)).T
			assert(my_z[k].shape == (2,1))


			#xhat[k] = xhatminus[k] + K[k]*(xhatminus[k]-C*xhatminus[k])
		else:
			#xhat[k] = xhatminus[k] + K[k]*(z[k]-C*xhatminus[k])
			my_z[k] = z[k]

		#print(K[k]*(my_z[k]-C*xhatminus[k]))
		xhat[k] = xhatminus[k] + K[k]*(my_z[k]-C*xhatminus[k])

		assert(xhat[k].shape == (4,1))

		P[k] = (I-K[k]*C)*Pminus[k]

		assert(P[k].shape == (4,4))

		#print(xhat[k])

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

		return [math.sqrt((xEstimate - xTrue)**2 + (yEstimate - yTrue)**2), (xEstimate - xTrue), (yEstimate - yTrue)]

	def computeMeasurementError(truth, measurement):
		assert(truth.shape == (4,1))
		assert(measurement.shape == (2,1))

		xTrue = truth[0,0]
		yTrue = truth[2,0]
		xMeasurement = measurement[0,0]
		yMeasurement = measurement[1,0]

		return [math.sqrt((xMeasurement - xTrue)**2 + (yMeasurement - yTrue)**2), (xMeasurement - xTrue) , (yMeasurement - yTrue)]

	estimationError = np.array([computeEstimationError(trueTrajectoryMatrix[i].T, xhat[i]) for i in range(len(xhat))])
	estimationErrorXY = estimationError[:,0]
	estimationErrorX = estimationError[:,1]
	estimationErrorY = estimationError[:,2]

	measurementError = np.array([computeMeasurementError(trueTrajectoryMatrix[i].T, z[i]) for i in range(len(xhat))])
	measurementErrorXY = measurementError[:,0]
	measurementErrorX = measurementError[:,1]
	measurementErrorY = measurementError[:,2]


	if(graph):
		plt.figure()
		plt.plot(trueTrajectoryArray[:,0], trueTrajectoryArray[:,2])
		plt.plot(rawMeasurementsArray[:,0], rawMeasurementsArray[:,1])
		plt.plot(xhat_a[:,0], xhat_a[:,2])
		plt.xlabel("x position")
		plt.ylabel("y position")
		plt.title("Estimated Trajectory (red), Raw Measurements (green), True Trajectory (blue) on xy-plane")
		plt.savefig("2a_Estimated_Raw_True_Trajectories.png")


		plt.figure()
		plt.plot(list(range(len(estimationError))), np.zeros(len(estimationError)))
		plt.plot(list(range(len(estimationError))), estimationErrorXY, color = 'green')
		plt.plot(list(range(len(measurementError))), measurementErrorXY, color = 'red')
		plt.xlabel("Time (sec)")
		plt.ylabel("Euclidean Distance / Error (m)")
		plt.title("Estimation (green) and Measurement error (red) vs. Time (sec)")
		plt.savefig("2a_Estimation_Measurement_error.png")


		plt.figure()
		plt.plot(list(range(len(trueTrajectoryArray))), trueTrajectoryArray[:,1])
		plt.plot(list(range(len(trueTrajectoryArray))),  estimationErrorX)
		plt.plot(list(range(len(trueTrajectoryArray))),  measurementErrorX)
		plt.xlabel("Time (sec)")
		plt.ylabel("meters")
		plt.title("Velocity on x (blue) with Estimation (green) and Measurement Error (red) vs. Time (sec)")
		plt.savefig("2a_velocity_along_x_est_err.png")


		plt.figure()
		plt.plot(list(range(len(trueTrajectoryArray))), trueTrajectoryArray[:,3])
		plt.plot(list(range(len(trueTrajectoryArray))),  estimationErrorY)
		plt.plot(list(range(len(trueTrajectoryArray))),  measurementErrorY)
		plt.xlabel("Time (sec)")
		plt.ylabel("meters")
		plt.title("Velocity on y (blue) with Estimation (green) and Measurement Error (red) vs. Time (sec)")
		plt.savefig("2a_velocity_along_y_est_err.png")

		# plt.show()

	return estimationErrorXY.sum() / measurementErrorXY.sum()




if __name__ == "__main__":
	# errorRatio = kalmanFilter(graph=True)
	# print(errorRatio)

	# 2b) try some variance_w to see the effects 
	# variance_w_list=[.003, .03, .3, 3, 30, 300]
	# for variance in variance_w_list:
	# 	errorRatio = kalmanFilter(variance_w=variance, graph=False)
	# 	print('For variance_w {} the error ratio is {}'.format(variance, errorRatio))

	# errorRatio = kalmanFilter(variance_w=3000)

	# 2c) try a very small variance_v
	# this made the model to assume a smaller variance_v 
	# errorRatio = kalmanFilter(variance_v = 1)

	# 2d) try variance_v = 1, variance_w = 1, initialState = [0, -100, 0, -100]
	# errorRatio = kalmanFilter(variance_v = 1, variance_w = 1, initialState = [0,-100,0,-100])

	# 2e) try poor sensorProbability = 0.5, sensor fails so often that sensor coudn't track correctly unless very little change/ no change in position. Errors are magnified when x or y moves, and it takes a lot longer than before until tracking goes back on right track
	errorRatio = kalmanFilter(sensorProbability = 0.5)
	errorRatio = kalmanFilter(sensorProbability = 1.0)


	plt.show()





