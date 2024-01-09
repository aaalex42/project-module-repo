"""
Stochastic support functions
"""

def round_to(val, rbase) -> float:
    """
    self explanatory, rounts val to rbase, result can be float
    """
    assert rbase != 0.0, 'rbase must not be 0'
    return round(val / rbase) * rbase

import random
import numpy

def lognorm(mu, varcoef) -> float:
    """
    scales the target expected value mu and varcoef = sigma/mu to the 
    embedded normal distribution with mu0 and sigma0 and returns
    the corresponding output
    beauty of lognorm: is always > 0 
    """
    mu0 = numpy.log(mu / numpy.sqrt(numpy.power(varcoef, 2) + 1.0))
    sigma0 = numpy.sqrt(numpy.log(numpy.power(varcoef, 2) + 1.0))
    return random.lognormvariate(mu0, sigma0)

def lognorm_int(mu, varcoef, rbase) -> int:
    """
    integer variant of lognorm to base rbase
    """
    return round(round_to(lognorm(mu, varcoef), rbase))

"""
Validation: 
mu = 10000.0
vc = 1.0
res = [lognorm_int(mu, vc, 100.0) for _ in range(10000)]
print(f'avg = {sum(res)/len(res)}')
"""
  
def uniform(mu, varcoef) -> float:
    """
    scales the target expected value mu and varcoef = sigma/mu to the 
    uniform distribution parameters [a, b] and returns
    the corresponding output
    """    
    a = (1.0 - numpy.sqrt(3.0) * varcoef) * mu
    b = (1.0 + numpy.sqrt(3.0) * varcoef) * mu
    assert a > 0, 'sqrt(3) * varcoeff must be < 1'
    return numpy.random.uniform(a, b)

def uniform_int(mu, varcoef, rbase) -> int:
    """
    integer variant of uniform to base rbase
    """
    return round(round_to(uniform(mu, varcoef), rbase))

def poisson(mu) -> float:
    """
    returns the poisson value for mu
    """    
    return numpy.random.poisson(mu)

def poisson_int(mu, rbase) -> int:
    """
    integer variant of poisson to base rbase
    """
    return round(round_to(poisson(mu), rbase))

"""
Validation: 
mu = 2.0
res = [poisson_int(mu, 1.0) for _ in range(10000)]
print(f'avg = {sum(res)/len(res)}')
"""

def distri_int(mu, varcoef, rbase) -> int:
    """
    wrapper for distribution forms
    supports lognormand uniform distribution 
    """
    if DISTRI_FORM == 'Lognorm':
        return lognorm_int(mu, varcoef, rbase)
    elif DISTRI_FORM == 'Uniform':
        return uniform_int(mu, varcoef, rbase)
    else:
        raise ValueError('{DISTRI_FORM} not defined')

"""
Validation: 
mu = 2000000.0
vc = 1.0
res = [distri_int(mu, vc, 1.0) for _ in range(10000)]
print(f'avg = {sum(res)/len(res)}')
"""

def distri(mu, varcoef) -> float:
    """
    wrapper for distribution forms
    comments see aove
    """
    if DISTRI_FORM == 'Lognorm':
        return lognorm(mu, varcoef)
    elif DISTRI_FORM == 'Uniform':
        return uniform(mu, varcoef)
    else:
        raise ValueError('{DISTRI_FORM} not defined')


def gen_demands(d_mu, d_varcoef, d_t_mu, rbase, nextd, curtd, amount):
    """
    generates a demand stream of length amount 
    
    returns nextd, curtd, demands
    
    d_mu, d_varcoef: demand distri params for amplitude 
    d_t_mu: demand distri param for time
    rbase: rounding base
    nextd: next time step for demand from here, is input and output
    curtd: current time step from last demand, is input and output
    amount: number of days, one element per day
    
    6-7-23: created this function, fixed curtd counter to start at 1 insread of 0
    """
    new_demands = []
    for _ in range(amount):
        if curtd >= nextd:
            new_demands.append(distri_int(d_mu, d_varcoef, rbase))
            curtd = 1
            nextd = poisson_int(d_t_mu, 1.0)
        else:
            new_demands.append(0)
            curtd += 1
    return curtd, nextd, new_demands
    


def gen_fc(d, d_fc_noise, d_mu, d_varcoef, rbase):
    """
    generates a forecast for the demand vector d
    returns the forecast vector which is of same length as the demand vector
    
    d_fc_noise: noise level in ratio 
    d_mu: mu of demand distribution
    d_varcoef: var coef of demand distribution
    rbase: rounding base
 
    6-7-23: created this function, rewrote the code, not it miraculously works somehow!
    11-21-23: minor, but dramatic little bug fix with carryover effect, NOW it
              miraculously works!
    """
    forecasts = []
    # the problem is truncation to positive forecast numbers. to do so and preserve the
    # stochastic qualities, noise_carryover stores the trunc'd values and adds them later on
    trunc_carryover = 0
    for di in d:
        noise = round_to(d_fc_noise * (distri_int(d_mu, d_varcoef, 1.0) - d_mu), rbase) 
        fci = max(0, di + noise + trunc_carryover)
        trunc_carryover = min(0, di + noise + trunc_carryover)
        forecasts.append(fci)
    return forecasts

"""
Validation:
dmu = 1000.0 # 1000 pieces with a variation of ...
vc = 0.5     # vc = 0.5 ...
dt = 3.0     # ... every 3 days and ...
rb = 10      # ... a demand rounding base of 100 pcs.
noise = 1.0  # fc noise level in vc
_, _, d = gen_demands(dmu, vc, dt, rb, 0, 0, 20000)
print(f'avg d = {sum(d)/len(d)}, should be {dmu/dt}')
n_0 = 0
n_not_0 = 0
for di in d:
    if di == 0:
        n_0 += 1
    else:
        n_not_0 += 1
print(f'dt actual = {n_0 / (n_not_0 - 1) + 1}, should be {dt}')
print(f'dmu actual = {sum(d) / n_not_0}, should b {dmu}')
f = gen_fc(d, noise, dmu, vc, rb)
print(f'avg f = {sum(f)/len(f)}, should be {dmu/dt}')
"""