#source: https://github.com/sagarvegad/Adam-optimizer
import math
import numpy as np

alpha = 0.01
beta_1 = 0.9
beta_2 = 0.999						#initialize the values of the parameters
epsilon = 1

def func(x):
	return np.sqrt(np.sum(np.square(x)))
def grad_func(x):					#calculates the gradient
	root = np.sqrt(np.sum(np.square(x)))
	return np.array([x[0] / root, x[1] / root])

theta_0 = np.array([10, 10])					#initialize the vector
m_t = np.array([0, 0])
v_t = np.array([0, 0])
t = 0

while (t < 10000):					#till it gets converged
	t+=1
	g_t = grad_func(theta_0)		#computes the gradient of the stochastic function
	m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient
	v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient
	m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates
	v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates
	theta_0_prev = theta_0
	theta_0 = theta_0 - np.divide((alpha * m_cap), (np.sqrt(v_cap) + epsilon))	#updates the parameters
	if(t%1000 == 0): #every sth
		print("theta_0: " + str(theta_0))

print("theta_0: " + str(theta_0))
print("t: " + str(t))
