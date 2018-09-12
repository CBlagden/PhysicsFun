import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np


x = [0, 2, 4, 5, 6, 8, 10]
y = [0.5, 1.5, 2.5, 2.5, 2.5, 1.5, 0.5]

def graph(x, y, name, xlabel, ylabel):
	plt.figure(num=name)
	plt.plot(x, y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.margins(0)

pos_spl = scipy.interpolate.UnivariateSpline(x, y, k=4)
vel_spl = pos_spl.derivative()
acc_spl = vel_spl.derivative()

x_new = np.linspace(0, 10, num=100)
time = 'time(s)'

graph(x_new, pos_spl(x_new), 'Position', time, 'position (m)')
graph(x_new, vel_spl(x_new), 'Velocity', time, 'velocity (m/s)')
graph(x_new, acc_spl(x_new), 'Acceleration', time, 'acceleration (m/s^2)')

plt.show()

