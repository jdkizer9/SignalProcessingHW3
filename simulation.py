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
	A0x = 2
	P0y = groundTruth[baseK][2]
	V0y = groundTruth[baseK][3]
	A0y = 2

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

def plotGroundTruth():

	groundTruthData = generateGroundTruth()
	##print(groundTruthData) 

	#select column 0
	groundTruthPx = groundTruthData[:,0]
	
	#select column 2
	groundTruthPy = groundTruthData[:,2]

	print(groundTruthData[:,0])
	print(groundTruthData[:,[0,2]])


	plt.figure()

	plt.plot(groundTruthPx, groundTruthPy)
	plt.show()

	# pylab.figure()
	# pylab

	# pylab.plot(z,'k+',label='noisy measurements')
	# pylab.plot(xhat,'b-',label='a posteri estimate')
	# pylab.axhline(x,color='g',label='truth value')
	# pylab.legend()
	# pylab.xlabel('Iteration')
	# pylab.ylabel('Voltage')

def generateNoisyMeasurementsFromGroundTruthData(groundTruthData, mean=[0,0], covariance=[[1000,0],[0,1000]]):

	noisyPositionData = groundTruthData[:,[0,2]] + np.random.multivariate_normal(mean, covariance, len(groundTruthData))
	
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

	groundTruthData = generateGroundTruth()
	noisyData = generateNoisyMeasurementsFromGroundTruthData(groundTruthData)

	plt.figure()
	plt.plot(groundTruthData[:,0], groundTruthData[:,2])
	plt.plot(noisyData[:,0], noisyData[:,2])

	plt.figure()
	plt.plot(list(range(len(groundTruthData))), groundTruthData[:,1])
	plt.plot(list(range(len(groundTruthData))),  noisyData[:,1])

	plt.figure()
	plt.plot(list(range(len(groundTruthData))), groundTruthData[:,3])
	plt.plot(list(range(len(groundTruthData))),  noisyData[:,3])

	plt.show()



if __name__ == "__main__":
	plotGroundTruthAndNoisyMeasureMentData()


