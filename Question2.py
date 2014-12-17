import kalman
import matplotlib.pyplot as plt

if __name__ == "__main__":
  # 2a) plots
  errorRatio = kalman.kalmanFilter(graph=True, prefix='2a', variance_w=.3*.3)
  print(errorRatio)

  #2b)
  errorRatio = kalman.kalmanFilter(graph=True, prefix='2b', variance_w=3000*3000)
  print(errorRatio)

  # 2c) try a very small variance_v
  errorRatio = kalman.kalmanFilter(graph=True, prefix='2c',variance_v = 1000)
  print(errorRatio)

  # 2d) try variance_v = 1, variance_w = 1, initialState = [0, -100, 0, -100]
  errorRatio = kalman.kalmanFilter(graph=True, prefix='2d',initialState = [0,-100,0,-100] )
  print(errorRatio)
  errorRatio = kalman.kalmanFilter(graph=True, prefix='2d_identity_covariance',variance_v = 1, variance_w = 1,initialState = [0,-100,0,-100] )
  print(errorRatio)

  #2e) tune 2d) via  until it looks good
  errorRatio = kalman.kalmanFilter(graph=True, prefix='2e',variance_v = 1000*1000, variance_w = 3*3,initialState = [0,-100,0,-100] )
  print(errorRatio)

  #2f) assume faulty sensor
  errorRatio = kalman.kalmanFilter(graph=True, prefix='2f', sensorProbability = 0.25)
