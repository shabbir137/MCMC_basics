import numpy as np
import matplotlib.pyplot as plt #plotting package

#  Define the distribution to be sampled
def prob_dist1(x, p1, p2, A):
    """Gaussian distribution with mean mu, 
    Standard deviation sigma and 'amplitude' A
    (A = 1 gives normalized gaussian.)"""
    mu = p1
    sigma = p2
    chi_sq = 0.5*((x - mu)/sigma)**2
    f = A*np.exp(-1.0*chi_sq)/(np.sqrt(2.0*np.pi)*sigma)
    return f

def prob_dist2(x, p1, p2, A):
    """Sum of two Gaussian distribution with mean mu1 & mu2, 
    Standard deviation sigma1 & sigma2 and 'amplitudes' A1 & A2
    (e.g. A1 = 0.5 & A2 = 0.5 gives normalized gaussian.)"""
    A1 = 0.5
    mu1 = p1
    sigma1 = p2
    chi_sq1 = 0.5*((x - mu1)/sigma1)**2
    f1 = A1*np.exp(-1.0*chi_sq1)/(np.sqrt(2.0*np.pi)*sigma1)
    A2 = 0.5
    mu2 = p1 + 3*p2
    sigma2 = p2*0.5
    chi_sq2 = 0.5*((x - mu2)/sigma2)**2
    f2 = A2*np.exp(-1.0*chi_sq2)/(np.sqrt(2.0*np.pi)*sigma2)
    return f1 + f2

#parameter of the distribution to be sampled
p1 = 0.0
p2 = 1.0
A = 1.0 #for normalized gaussian distribution

fig = plt.figure(figsize=(8,6), dpi=100)
fig.add_subplot(111)
x = np.arange(-8*p2, 8*p2, 0.01)
plt.plot(x, prob_dist2(x, p1, p2, A), linewidth = 2.0, color = 'k')

#number of samples
nsample = 100

#Choose initial point to start the chain
theta_0 = 1.0
theta_i = theta_0

#Parameters of the proposal distribution
#In our example, standard deviation of the gaussian proposal distribution
sigma_p = 1.5

#Seed for random number generation while sampling
seed1 = 1012345
np.random.seed(seed = seed1)

#array to store total sample
total_sample = np.ndarray(shape = (nsample, 2))
#array to store accepted sample
acptd_sample = np.ndarray(shape = (nsample, 2))

#array to store the value of distribution function
f = np.ndarray(shape = (nsample, 1))

#Compute the function at the starting point
f[0] = prob_dist2(theta_i, p1, p2, A)

#Following loop does the Markov Chain Monte Carlo (MCMC) 
#sampling of the distribution.

n_accept=0
for i in range(1, nsample,1):
    #gaussian proposal distribution
    theta_star = np.random.normal(loc = theta_i, scale = sigma_p, size = 1)
    total_sample[i,:] = np.asarray([i, theta_star])
    #Plot verticle line at proposed point, with red color
    plt.vlines(theta_star, 0.0, 0.4, color = 'r', linewidth = 1.0)
    plt.pause(1.0)
    #Compute function at the proposed point
    f_star = prob_dist2(theta_star, p1, p2, A)
    #Metropolis rule
    if f_star > f[i-1]:
        #accept proposed point
        theta_i = theta_star
        acptd_sample[i,0] = i
        acptd_sample[i,1] = theta_i
        f[i] = f_star
        n_accept += 1
        plt.vlines(theta_star, 0.0, 0.5, color = 'g', linewidth = 1.0)
        plt.pause(1.0)
    else:
        alpha = np.random.uniform(low=0.0, high=1.0, size=None)
        ratio = f_star/f[i -1]
        if ratio > alpha:
            #accept proposed point
            theta_i = theta_star
            acptd_sample[i,:] = np.asarray([i, theta_i])
            f[i] = f_star
            n_accept += 1
            plt.vlines(theta_star, 0.0, 0.5, color = 'g', linewidth = 1.0)
            plt.pause(1.0)
        else:
            #reject proposed point
            theta_i = theta_i #chain stays at the currant point.
            #Currant (not the proposed) point is re-added to the accepted sample.
            acptd_sample[i,:] = np.asarray([i, theta_i])
            f[i] = f[i-1]
            plt.vlines(theta_i, 0.0, 0.5, color = 'g', linewidth = 1.0)
            plt.pause(1.0)
        print("acceptance ratio:")
        print(n_accept/(1.0*i))

