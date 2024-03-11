import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

def merton_jump_paths(S, T, r, sigma,  lam, m, v, steps, Npaths):
    size=(steps,Npaths)
    dt = T/steps 
    poi_rv = np.multiply(np.random.poisson( lam*dt, size=size),
                         np.random.normal(m,v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r -  sigma**2/2 -lam*(m  + v**2*0.5))*dt +\
                              sigma*np.sqrt(dt) * \
                              np.random.normal(size=size)), axis=0)
    
    return np.exp(geo+poi_rv)*S


S = 116.75 # current stock price
T = 1 # time to maturity
r = 0.05 # risk free rate
m = 0 # meean of jump size
v = 0.1 #volatiliy of jump size
lam = 1 # intensity of jump i.e. number of jumps per annum
steps =255 # time steps
Npaths =2000 # number of paths to simulate
sigma = 0.3245 # annaul standard deviation , for weiner process
K =140
#np.random.seed(3)
j = merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths) #generate jump diffusion paths

plt.plot(j)
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Jump Diffusion Process')
#plt.show()




mcprice = np.maximum(j[-1]-K,0).mean() * np.exp(-r*T) # calculate value of call

print('Monte Carlo Merton Price =', mcprice)
