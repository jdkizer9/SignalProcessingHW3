import numpy as np
import matplotlib.pyplot as plt

#returns numpy array with ground truth state vectors for each point in time k
def generateGroundTruth():

	groundTruth = np.zeros((501,4))

	##[px, vx, py, vy]
	groundTruth[0] = np.array([0, 300, 0, 0])

	baseK = 0
	P0x = groundTruth[baseK][0]
	V0x = groundTruth[baseK][1]
	A0x = 0
	P0y = groundTruth[baseK][2]
	V0y = groundTruth[baseK][3]
	A0y = 0

	for k in range (1, 201):
		##[px, vx, py, vy]
		px = .5*A0x*(k**2) + V0x*k + P0x
		vx = A0x*k + V0x
		py = .5*A0y*(k**2) + V0y*k + P0y
		vy = A0y*k + V0y
		groundTruth[k+baseK] = np.array([px, vx, py, vy]) 

	baseK = 200
	P0x = groundTruth[baseK][0]
	V0x = groundTruth[baseK][1]
	A0x = 2.
	P0y = groundTruth[baseK][2]
	V0y = groundTruth[baseK][3]
	A0y = 2.

	for k in range (1, 101):
		##[px, vx, py, vy]
		px = .5*A0x*(k**2) + V0x*k + P0x
		vx = A0x*k + V0x
		py = .5*A0y*(k**2) + V0y*k + P0y
		vy = A0y*k + V0y
		groundTruth[k+baseK] = np.array([px, vx, py, vy]) 

	#update vx(300) = 0 and vy(300) = 300
	baseK = 300
	groundTruth[baseK][1] = 0
	groundTruth[baseK][3] = 300

	P0x = groundTruth[baseK][0]
	V0x = groundTruth[baseK][1]
	A0x = 0
	P0y = groundTruth[baseK][2]
	V0y = groundTruth[baseK][3]
	A0y = 0

	for k in range (1, 201):
		##[px, vx, py, vy]
		px = .5*A0x*(k**2) + V0x*k + P0x
		vx = A0x*k + V0x
		py = .5*A0y*(k**2) + V0y*k + P0y
		vy = A0y*k + V0y
		groundTruth[k+baseK] = np.array([px, vx, py, vy]) 

	return groundTruth

def generateDataWithMatrices(T=1., measurementNoise=([0,0], [[1000,0],[0,1000]])):

	##len 501 array of 4x1 Matrices


	xhat = np.empty(501, dtype=np.dtype(object))
	xhatminus = np.empty(501)

	xhat[0] = np.matrix([0., 300., 0., 0.])
	#print(xhat[0])

	A = np.matrix([[1., T, 0., 0.], [0., 1., 0., 0.], [0., 0., 1., T], [0., 0., 0., 1.]])
	B = np.matrix([[.5*T**2, 0], [T, 0.], [0., .5*T**2], [0, T]])

	u = np.matrix([0., 0.])

	for i in range(1, 201):
		#print(A)
		#print(xhat[i-1].T)
		xhat[i] = (A*xhat[i-1].T + B*u.T).T
		#print(xhat[i])


	u = np.matrix([2., 2.])
	for i in range(201, 301):
		xhat[i] = (A*xhat[i-1].T + B*u.T).T

	# print(xhat[200])
	# print(A*xhat[200].T)
	# print(B*u.T)
	# print(xhat[201])

	u = np.matrix([0., 0.])
	# print(xhat[300])
	# print(type(xhat[300]))
	xhat[300][0,1] = 0
	xhat[300][0,3] = 300
	for i in range(301, 501):
		xhat[i] = (A*xhat[i-1].T + B*u.T).T


	z = np.empty(len(xhat), dtype=np.dtype(object))
	z[0] = np.matrix([xhat[0][0,0], xhat[0][0,2]])
	C = np.matrix([[1., 0., 0., 0.], [0., 0., 1., 0.]])
	for i in range(1, len(xhat)):
		#print(C.shape)
		#print(xhat[i].T.shape)
		#print(C*xhat[i].T)
		#print(np.matrix(np.random.multivariate_normal(*measurementNoise)).T)
		z[i] = (C*xhat[i].T + np.matrix(np.random.multivariate_normal(*measurementNoise)).T).T
		#print(z[i].shape)


	z_velocity = np.empty(len(z), dtype=np.dtype(object))
	z_velocity[0] = np.matrix([z[0][0,0], 0., z[0][0,1], 0.])
	for i in range(1, len(z)):
		z_velocity[i] = np.matrix([z[i][0,0], z[i][0,0] - z[i-1][0,0], z[i][0,1], z[i][0,1] - z[i-1][0,1]])
		# z_velocity[i][0,0] = z[i][0,0]
		# z_velocity[i][0,1] = z[i][0] - z[i-1][0]
		# z_velocity[i][0,2] = z[i][0,1]
		# z_velocity[i][0,3] = z[i][0,1] - z[i-1][0,1]

	xhat_a = np.zeros((len(xhat),4))
	for i in range(len(xhat)):
		#print(xhat[0].shape)
		xhat_a[i] = xhat[i].getA()

	print(len(z))
	z_a = np.zeros((len(z),2))
	for i in range(len(z)):
		#print(z[i])
		#print(z[i].shape)
		z_a[i] = z[i].getA()

	return xhat_a, z_a, xhat, z, z_velocity

def plotGroundTruth():

	groundTruthData = generateGroundTruth()
	groundTruthData_m, measuredData = generateDataWithMatrices()
	groundTruthData_mo = np.zeros((len(groundTruthData_m),4))
	for i in range(len(groundTruthData_m)):
		groundTruthData_mo[i] = groundTruthData_m[i].getA()

	##print(groundTruthData) 

	#select column 0
	groundTruthPx = groundTruthData[:,0]
	
	#select column 2
	groundTruthPy = groundTruthData[:,2]

	# print(groundTruthData[:,0])
	# print(groundTruthData[:,[0,2]])


	plt.figure()

	#print(groundTruthData)
	#print(groundTruthData_m)
	plt.plot(groundTruthData[:,0], groundTruthData[:,2])
	plt.plot(groundTruthData_mo[:,0], groundTruthData_mo[:,2])
	plt.show()

	# pylab.figure()
	# pylab

	# pylab.plot(z,'k+',label='noisy measurements')
	# pylab.plot(xhat,'b-',label='a posteri estimate')
	# pylab.axhline(x,color='g',label='truth value')
	# pylab.legend()
	# pylab.xlabel('Iteration')
	# pylab.ylabel('Voltage')

def generateNoisyMeasurementsFromGroundTruthData(groundTruthData, measurementNoise=([0,0], [[1000,0],[0,1000]])):

	noisyPositionData = groundTruthData[:,[0,2]] + np.random.multivariate_normal(measurementNoise[0], measurementNoise[1], len(groundTruthData))
	
	noisyMeasurement = np.zeros((501,4))
	for i, noisyPosition in enumerate(noisyPositionData[0:len(groundTruthData)-1,:]):
		noisyMeasurement[i][0] = noisyPositionData[i][0]
		noisyMeasurement[i][1] = noisyPositionData[i+1][0] - noisyPositionData[i][0]
		noisyMeasurement[i][2] = noisyPositionData[i][1]
		noisyMeasurement[i][3] = noisyPositionData[i+1][1] - noisyPositionData[i][1]

	noisyMeasurement[len(groundTruthData)-1][0] = noisyPositionData[len(groundTruthData)-1][0]
	noisyMeasurement[len(groundTruthData)-1][2] = noisyPositionData[len(groundTruthData)-1][1]

	return noisyMeasurement

def plotGroundTruthAndNoisyMeasureMentData():

	# groundTruthData = generateGroundTruth()
	# noisyData = generateNoisyMeasurementsFromGroundTruthData(groundTruthData)

	groundTruthData, noisyPositionData, xhat, z, z_velocity = generateDataWithMatrices()


	#print(noisyPositionData)

	noisyMeasurement = np.zeros((501,4))
	for i, noisyPosition in enumerate(noisyPositionData[0:len(groundTruthData)-1,:]):
		noisyMeasurement[i][0] = noisyPositionData[i][0]
		noisyMeasurement[i][1] = noisyPositionData[i+1][0] - noisyPositionData[i][0]
		noisyMeasurement[i][2] = noisyPositionData[i][1]
		noisyMeasurement[i][3] = noisyPositionData[i+1][1] - noisyPositionData[i][1]

	noisyMeasurement[len(groundTruthData)-1][0] = noisyPositionData[len(groundTruthData)-1][0]
	noisyMeasurement[len(groundTruthData)-1][2] = noisyPositionData[len(groundTruthData)-1][1]


	plt.figure()
	plt.plot(groundTruthData[:,0], groundTruthData[:,2])
	# plt.plot(noisyMeasurement[:,0], noisyMeasurement[:,2])
	plt.xlabel("x position")
	plt.ylabel("y position")
	plt.title("Exact Trajectory")
	plt.axis([0,120000,-10000,80000])
	plt.savefig("1b_groundTruth_trajectory.png")


	plt.figure()
	plt.plot(groundTruthData[:,0], groundTruthData[:,2])
	plt.plot(noisyMeasurement[:,0], noisyMeasurement[:,2])
	plt.xlabel("x position")
	plt.ylabel("y position")
	plt.title("Exact Trajectory with Noisy Measurement")
	plt.savefig("1c_groundTruth_trajectory_with_noisy_measurement.png")

	plt.figure()
	plt.plot(list(range(len(groundTruthData))), groundTruthData[:,1])
	# plt.plot(list(range(len(groundTruthData))),  noisyMeasurement[:,1])
	plt.xlabel("time (sec)")
	plt.ylabel("x velocity")
	plt.title("Velocity along X-axis")
	# set min max on axis
	plt.axis([0,500,-100,600])
	plt.savefig("1d_velocity_along_x.png")


	plt.figure()
	plt.plot(list(range(len(groundTruthData))), groundTruthData[:,3])
	# plt.plot(list(range(len(groundTruthData))),  noisyMeasurement[:,3])
	plt.xlabel("time (sec)")
	plt.ylabel("y velocity")
	plt.title("Velocity along Y-axis")
	# set min max on axis
	plt.axis([0,500,-100,400])
	plt.savefig("1d_velocity_along_y.png")

	plt.show()



if __name__ == "__main__":
	# plotGroundTruth()
	plotGroundTruthAndNoisyMeasureMentData()


