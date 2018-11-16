from math import cos, radians, hypot
import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np
from sklearn import linear_model
import numpy as np

def graph(x, y, name='', xlabel='', ylabel=''):
	plt.figure(num=name)
	plt.plot(x, y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

h_initial = 0.273 # m
spring_compression = 0.035 # m
g = 9.81 # m/s^2
m = 0.01 # kg

angles = [[15, 30, 45, 60]] # degrees
distance = [0.965, 1.55, 1.82, 1.56] # m
time = [0.77, 0.69, 0.63, 0.62] # s

k_data = [[]]

for t1, x, theta in zip(time, distance, angles[0]):
	vel_i_x = x / t1
	vel_i_y = (0.5 * g * t1 * t1) / t1
	t2 = 2 * vel_i_y / g
	t_vertex = t2 / 2

	h = vel_i_y * t_vertex - 0.5 * g * t_vertex * t_vertex
	print(h)
	k = (m * (vel_i_x)**2 + 2 * m * g * h) / (spring_compression**2)

	k_data[0].append(k)

regr = linear_model.LinearRegression()
regr.fit(angles, k_data)

k = np.mean(k_data[0])
print("k = %f" % k)

# angles = [np.arange(0, 90, 0.1)]
# print(regr.predict([angles]))

graph(angles, regr.predict(angles), name="angle vs sping constant", xlabel="angle", ylabel="k")

plt.scatter(angles, regr.predict(angles))
plt.show()

