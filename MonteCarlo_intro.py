import numpy as np
import matplotlib.pyplot as plt

nsample = 500 #number of samples
#Seed for random number generation while sampling
seed1 = 1012345 #Seed for the random number generator
np.random.seed(seed = seed1) #Seed the random number generator
x = np.random.uniform(low=-1.0, high=1.0, size=nsample)
seed2 = 10145
np.random.seed(seed = seed2)
y = np.random.uniform(low=-1.0, high=1.0, size=nsample)

#Plot histogram of x values
plt.hist(x, normed=True, bins=10)
plt.show()

#Plot x
plt.plot(x)
plt.show()

#Compute the autocorrelation of 'x' chain, to check the Markovian nature
max_lag = 50

lag = np.arange(0,max_lag) # array for correlation length
cov = np.ndarray(shape = (max_lag,1)) # array for correlation

mean_x = np.mean(x) #mean of x
var_x = np.var(x) # variance of x

#Compute the autocorrelation of 'x' chain
for l in range(0,max_lag):
    sum1 = 0.0
    for i in range(0, len(x)-max_lag):
        sum1 = sum1 + (x[i] - mean_x)*(x[i+l] - mean_x)
    cov[l] = sum1/((len(x) - max_lag)*var_x) 

#Plot the correlation length
plt.plot(lag, cov)

horizontal_line = np.ndarray(shape = (max_lag, 1))
horizontal_line.fill(0.0)
plt.plot(lag, horizontal_line, linewidth = 2.0, linestyle = '--', c = 'k')

plt.show()

plt.scatter(x, y)
plt.show()

np.cov(x,y)

#Simple Example of Markov Chain Monte Carlo method 
#Estimate value of 'pi'
num_pair_in_circle = 0
for i in range(0, nsample, 1):
    x1 = x[i]; y1 = y[i]
    r = np.sqrt(x1**2 + y1**2)
    #Count the number of (x,y) inside the unit circle
    if (r < 1.0):
        num_pair_in_circle = num_pair_in_circle + 1

piby4 = 1.0*num_pair_in_circle/nsample #(area of circle)/(area of square) = pi/4

rel_diff = (4.0*piby4 - np.pi)/np.pi

print(rel_diff)

