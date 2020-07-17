import scipy.integrate as integrate
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import math

ex = 1.045
sa = 12.10

ex_2 = 3.90
sa_2 = 13.10

def expected_Earnigs(mean, std_v, ordered):
    return integrate.quad(lambda x: ((ex + sa)*x - ex*ordered)*norm.pdf(x, mean, std_v),0,ordered) + integrate.quad(lambda x: ex*ordered*norm.pdf(x, mean, std_v),ordered,100000)

#data is in format (mean, std_v)
D = [(30763, 13843),
    (10569,4756), 
    (8159,3671),
    (7270,4362),
    (5526,3316),
    (2118,1271)]



def objective_func(x):
    sum = 0.0
    for idx in range(6):
        
        sum += expected_Earnigs(D[idx][0], D[idx][1], x[idx])[0]

    return -sum

bounds =  [(0, 1000000) for idx in range(6)]

res = minimize(objective_func, x0 = 20000*np.ones(6), bounds = bounds)
print(res)