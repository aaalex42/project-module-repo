import numpy as np
import random
from main import DAYS_PER_YEAR, FOP_PERIODS, DAYS_PER_WEEK

"""
TODO:
    - Change the formulae for mu0 and sigma0 in lognorm_int function

Changes:
    - 11.12.23: functions eoq_static and eoq_dynamic are moved here
"""

def round_to(x, base):
    """
    rounds x to the next base
    """
    assert base > 0, "base must be greater than 0"

    return base * np.round(x/base)


def lognorm_int(mu, varcoef, rbase=1, size=1, round=True) -> int:
    """
    returns the lognormal value 
    scales the target expected value mu and varcoef = sigma/mu to the 
        embedded normal distribution with mu0 and sigma0 and returns
        the corresponding output
    beauty of lognorm: is always > 0

    rounds it to an integer
    """

    mu0 = np.log(mu / np.sqrt(varcoef ** 2 + 1.0))
    sigma0 = np.sqrt(np.log(varcoef ** 2 + 1.0))
    
    x = np.random.lognormal(mu0, sigma0, size = size)

    #return random.lognormvariate(mu0, sigma0)
    if round:
        return round_to(np.round(x), rbase)
    else:
        return x[0]


def uniform_int(mu, varcoef, rbase, size = 1) -> int:
    """
    returns the uniform value
    scales the target expected value mu and varcoef = sigma/mu to the 
        uniform distribution parameters [a, b] and returns
        the corresponding output

    rounds it to an integer
    """

    a = (1.0 - np.sqrt(3.0) * varcoef) * mu
    b = (1.0 + np.sqrt(3.0) * varcoef) * mu

    assert a > 0, 'sqrt(3) * varcoef must be < 1'

    x = round_to(np.random.uniform(a, b, size = size), rbase)

    return np.round(x)


def poisson_int(mu, rbase, size=1) -> int:
    """
    returns the poisson value for mu

    rounds it to an integer
    """
    x = round_to(np.random.poisson(mu, size = size), rbase)

    return np.round(x)


def random_dist_int(dist_type, mu, varcoef, rbase, size = 1) -> int:
    """
    returns the random value for the specified distribution
    """
    # for better checking
    dist_type = dist_type.lower()

    if dist_type == "lognorm":
        return lognorm_int(mu, varcoef, rbase, size=size)
    elif dist_type == "uniform":
        return uniform_int(mu, varcoef, rbase, size=size)
    elif dist_type == "poisson":
        return poisson_int(mu, rbase, size=size)
    else:
        raise ValueError("{dist_type} type not supported")


"""
Now creating vectorized functions for the above function to improve performance
"""

def gen_demands(d_mu, d_varcoef, d_t_mu, fcast_period):
    """
    This is a vectorized function that creates a demands list for a given forecast period
    """
    # generate a list of poisson values as with mu = d_t_mu and round it to int
    poisson_idx = np.round(np.random.poisson(d_t_mu, fcast_period))
    # calculate the cumulative sum to have it as indices and then filter out the ones with are grater than the forecast period
    poisson_idx = np.cumsum(poisson_idx)
    poisson_idx = poisson_idx[poisson_idx < fcast_period]

    # generate an array of zeros of demands
    demands = np.zeros(fcast_period)
    # fill the demands with the poisson values
    demands[poisson_idx] = lognorm_int(d_mu, d_varcoef, size = len(poisson_idx))

    return demands

# FUNCTION gen_fc IS NOT IMPLEMENTED AS A VECTORIZED ONE, BECAUSE IT IS NOT USED IN THE CODE LATER AND IT IS NOT RELEVANT FOR THE EXPERIMENT


"""
-------------------------------
"""

def eoq_static(J, E_p, B_k, Z):
    """
    This function calculates the EOQ with Andler formula.
    This function assumes constant demand, which is not true in most of the cases

    J   - annual demand
    E_p - unit cost
    B_k - setup cost per batch
    Z   - interest per year for storage, capital, etc.
    """

    return np.round(np.sqrt( (2 * J * B_k) / (E_p * Z)))


def eoq_dynamic(demand, E_p, B_k, Z, mode):
    """
    This function calculates the EOQ with different methods.

    demand - list of demands
    E_p    - unit cost
    B_k    - setup cost per batch
    Z      - interest per year for storage, capital, etc.  
    mode   - "Andler", "FOP" and "FOW

    Changes:
        - 12.12.23: round to int for FOW mode (Andler mode done in eoq_static)

    Comments from last version:
        20.06.23 - fix for FOP: from demand[0:x-1] to demand[0:x]
        21.11.23 - added FOW for four weeks of material, it is, however, identical to FOP = 28
    """
    # change mode to lower case
    mode = mode.lower()

    if mode == "andler":
        # This is an incorrect implementation, because the demand is not constant
        J = np.round( (np.sum(demand) / len(demand)) * DAYS_PER_YEAR )
        return eoq_static(J, E_p, B_k, Z)
    
    elif mode == "fop":
        # fixed order period
        return np.sum(demand[: np.min(FOP_PERIODS, len(demand)) ])
    
    elif mode == "fow":
        # four-week method used by ITW: add 4 weeks of stock
        # calculate the average demand per day and return 4 weeks of demand
        # round to an integer
        return np.round(np.mean(demand) * DAYS_PER_WEEK * 4)
    
    else:
        raise ValueError(f"mode {mode} not supported")
    



