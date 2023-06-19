# -*- coding: utf-8 -*-
"""
Simulates a plant with one machine and two products
Implements:
    - demand and forecast generation
    - demand fulfillment from stock
    - stock reorder point observation and control
    - dynamic economic order quantity calculaion
    - sequence scheduling

Limitations:
- The time unit is 1 for one day. First day is 0.
- Rough and ready, some of the patterns have not been tested yet
- Currently limited to 1 order per product per day, bucket is also 1 day
- no machine breakdowns implemented yet

@author: ruekgauer

Changelog:
    V 1.0 17.4.23 Initial external version
          24.4.23 minor adjustment: OEE is w/o setup time
    V 1.1  2.5.23 changes:
                  Inventory init uses the very regime later required for sim
                  at startup, an expected OEE is calculated
    V 2.0 16.5.23 Demand determination changed to compounded Poisson
                  process which better reflects reality. This requires major
                  adjustments, therefore major revision change
                  added demand and fulfillment tracking
                  
"""



"""
Time and date definitions and conventions used herein
"""
DAYS_PER_WEEK = 7
WEEKS_PER_YEAR = 52
DAYS_PER_YEAR = WEEKS_PER_YEAR * DAYS_PER_WEEK
HOURS_PER_DAY = 24
SEC_PER_HOUR = 60 * 60
SEC_PER_DAY = SEC_PER_HOUR * HOURS_PER_DAY
# this is the central scaling factor for the simulation: 
ONE_DAY = 1.0
FIFTEEN_MINUTES = ONE_DAY / HOURS_PER_DAY / 60 * 15

"""
Definition of start and end of simulation
"""
SIM_CYCLES = 365
SIM_START = 0.0
SIM_END = SIM_CYCLES * ONE_DAY

"""
Process specific definitions
"""
# distribution form for all stochastic data, see options below:
DISTRI_FORM = 'Lognorm'
# can or cannot provide partial shipments:
ALLOW_PART_SHIPMENTS = True 
# rounding base for demands and forecasts:
RBASE = 100
# size of the forecast for all products:
DAYS_FC = 180
# span in periods used in case of fixed order period quantity determination:
FOP_PERIODS = 50
# var coef for machine execution per propdution step:
MACHINE_VARCOEF = 0.1
# scheduling regime used herein, see defitions below:
SCHEDULING_REGIME = 'SPT'

"""
We are dealing with 2 products herein, their corresponding economic framework 
defined as follows

The demand is double stochastic: a Poisson time distance, and a (not quite) Lognorm amount variation

1. Niche product:
    avg demand: 12.000 pcs (when only real demand days are analyzed which is a subset of all data mostly filled with 0)
        stdev: 10.000 pcs
    avg time step: 16 days
        stdev: 17 days

2. Volume product: 
    avg demand: 140.000 pcs
        stdev: 300.000 pcs
    avg time step: 12 days
        stdev: 10 days
        
The real machine OEE is anywhere between 69% and 82%
"""
Q_PER_DAY = 12000 # product 1 is low amount, high volatility, high uncertainty
p1 = { \
    'd_mu'         : Q_PER_DAY, # number of pieces per day produced \
    'd_varcoef'    : 10/12, # var coef for demand determination \
    'd_t_mu'       : 16, # number of days on avg between 2 orders \
    'd_fc_noise'   : 1.0, # var coef of forecast noise 0...1 as percentage \
    'safety_stock' : 4.0 * Q_PER_DAY * DAYS_PER_WEEK, # safety stock: 4 weeks \
    'E_p'          : 0.03,  # cost per piece for economic order quantity calc \
    'B_k'          : 200,  # setup cost per batch for eoq calc \
    'irate'        : 0.1,  # inventory interest for eoq calc \
    't_e'          : 5.0 / SEC_PER_DAY,  # production time per piece in fractions of a day \
    't_r'          : 2.0 / HOURS_PER_DAY,  # setup time per batch in fractions of a day \
    'eoq_mode'     : 'Andler' # which eoq model to use, definitions see below \
}
    
Q_PER_DAY = 140000 # product 2 is high amount, low volatility, medium uncertainty
p2 = { \
    'd_mu'         : Q_PER_DAY, \
    'd_varcoef'    : 30/14,  \
    'd_t_mu'       : 12, \
    'd_fc_noise'   : 0.5, \
    'safety_stock' : 4.0 * Q_PER_DAY * DAYS_PER_WEEK, \
    'E_p'          : 0.04, \
    'B_k'          : 300,  \
    'irate'        : 0.1,  \
    't_e'          : 2.5 / SEC_PER_DAY, \
    't_r'          : 3.0 / HOURS_PER_DAY, \
    'eoq_mode'     : 'FOP' \
}
 
master_product_data = [p1, p2]

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
    

class Demand():
    """
    demand class, implements 
    - demand generation
    - forecast generation
    - fulfillment and backlog handling
    - statical fulfillment analysis
    by definition, one demand per day is assumed. Whenever demand is requested,
    it will be processed and new demand for the next day is generated, as well 
    as forecast is updated
    
    actual future demand stream is stored internally for consistent generation of forecasts
    definition: 
    - demands[0] is current. new demand is added at the end. demand[0] is 
      removed from the front after fulfillment attempt), 
    - demands[1:d_fc] are future demands used for forecasting
    
    Demand is always for 1 product
    
    from V1.1 it is essential for Demand.init to setup a full demand stream as
    it used for Inventory.init
    """
    def __init__(self, p, rbase, d_fc, env):
        """
        inits everything and generates an initial demand pattern stream
        p: product
        rbase: rounding base
        d_fc: days forecast
        """
        self.p = p            # remembers product
        self.rbase = rbase    # rounding base
        self.n_good = 0       # counts the successful demand fulfillments
        self.n_bad = 0        # counts the unsuccessful ones (not on time or amount)
        self.backlog = 0      # stores the unfulfilled demands for later
        self.demands = []     # holds the actual demands[d_fc], subject, is a ring buffer
        self.forecasts = []   # holds the noised up forcasts [d_fc, first entry is tomorrow]
        self.nextd = 0        # next event in days where demand is generated
        self.curtd = 0        # pointer to current time relative to nextd
        self.t = []           # time stamp for fulfillment, just for analysis 
        self.ff = []          # fulfilled amount
        self.d = []           # demanded amount
        self.delta_bl = []    # delta backlog
        self.env = env        # dto, just for analysis puposes

        # initially build demand stream, 1st is current demand
        self.nextd = poisson_int(p['d_t_mu'], 1) 
        self.gen_demand_and_fc(d_fc + 1) 
    
    def gen_demand_and_fc(self, amount):
        """
        generates new demand, simply a cast
        important: appends at the end (fulfill will purge at the front)
        and updates fc for demands other than current (1st in list)
        """
        
        for _ in range(amount):
            if self.curtd == self.nextd:
                self.demands.append(distri_int(self.p['d_mu'], self.p['d_varcoef'], self.rbase))
                self.curtd = 0
                self.nextd = poisson_int(self.p['d_t_mu'], 1)
            else:
                self.demands.append(0)
                self.curtd += 1
        
        self.forecasts = []
        d = self.demands[1:-1]
        for i in range(len(d)):
            self.forecasts.append(max(0, round_to(d[i] + self.p['d_fc_noise'] \
                                             * (distri_int(self.p['d_mu'], self.p['d_varcoef'], 1.0) \
                                                - self.p['d_mu'] / 2.0), self.rbase)))
            
    def fulfill(self, mat, allow_part_shipments) -> int:
        """
        mat : mateial amount handed down to manage demand stream
        allow_part_shipments : bool, allow for partial shipments or no
        return: material consumed for this step <= mat
        will remove demand (partially) fulfilled from front of demands list
        definition:
        - if demand cannot be fulfilled due to material constraints,
          backlog is built up
        - backlog is processed first, the current demand
        """
        mat_used = 0
        bl0 = self.backlog

        # 1. first handle backlog as much as possible
        if mat >= self.backlog:
            mat_used += self.backlog
            mat -= self.backlog
            self.backlog = 0
        else:
            mat_used += mat
            self.backlog -= mat
            mat = 0
        # 2. then handle current demand step as much as possible and update statistics
        if mat >= self.demands[0]:
            self.n_good += 1
            mat_used += self.demands[0]
        else:
            self.n_bad += 1
            if allow_part_shipments:
                self.backlog += (self.demands[0] - mat)
                mat_used += mat
                mat = 0
            else:
                self.backlog += self.demands[0]
        # 3. purge current demand from ringbuffer and save statistics
        self.t.append(self.env.now)
        self.d.append(self.demands[0])
        self.ff.append(mat_used)
        self.delta_bl.append(bl0 - self.backlog)
        del self.demands[0]
        return mat_used
     
    def service_level(self):
        """
        calculates the service level at any time defined as
        # good shipments (proper amount and time) / # total number of demands
        """
        return self.n_good / (self.n_good + self.n_bad)
    
    def avg_daily_machine_time(self):
        """
        returns the expected daily machine time consumed by the product
        new definition from V2.0 on: 
            daily demand = mu_d / d_t_mu
        """
        p = self.p
        daily_demand = p['d_mu'] / p['d_t_mu']
        daily_prod_time = daily_demand * p['t_e']
        eoq = eoq_dynamic(self.demands, p['E_p'], p['B_k'], p['irate'], p['eoq_mode'])
        daily_setup_time = p['d_mu'] / p['d_t_mu'] / eoq * p['t_r'] 
        return daily_prod_time + daily_setup_time

            
def eoq_static(J, E_p, B_k, Z) -> int:
    """ 
    implements the Andler formula 
    note the various imitations, in particular, constant demand pattern which 
    is most often violated herein
    J: annual demand
    E_p: unit cost
    B_k: setup cost per batch
    Z: interest per year for storage, capital, ...
    """
    return round(numpy.sqrt(2.0 * J * B_k / E_p / Z))

def eoq_dynamic(d, E_p, B_k, Z, mode) -> int:
    """ 
    calculcates the economic quanity, various models can be implemented 
    d: demand list
    mode: see below
    rest like eoq_static
    """
    if mode == 'Andler':
        # Andler, note this is not correct as the equation assumes constant demand
        sum_d = sum(d)
        len_d = len(d)
        J = round(sum_d / len_d * DAYS_PER_YEAR)
        return eoq_static(J, E_p, B_k, Z)
    elif mode == 'FOP':
        # fixed order period
        return sum(d[0:min(FOP_PERIODS, len(d)) - 1])
    else:
        raise ValueError('{mode} not defined')

class Inventory():
    """ 
    - passive device just holding inventory level and reorder point info 
      initial amount = safety + static eoq
    - reorder point also static for replenishemnt cycle based on eoq * t_e + t_r
    - also stores all past levels for later analysis both in time and amounts
    - this class also holds the information for replenishment handling: at the
      beginning of a replen cycle, replen_due and replen_open are set to the least 
      amount = rop - level, then this routine counts down replen_due and the
      fulfillment routine counts down replen_open
      
    Invenory is always for one product
    
    from V1.1 it uses the same quantity regime as planning uses
    """
    def __init__(self, p):
        """
        d: demand instance
        env: simpy environment, only needed for env.now (statisical purposes)
        """
        #p = d.p
        #self.env = env
        #self.t = [self.env.now]
        #eoq = eoq_dynamic([d.demands[0]] + d.forecasts, \
        #                  p['E_p'], p['B_k'], p['irate'], p['eoq_mode'])
        self.eoq = 10*p["d_mu"]
        self.levels = [p['safety_stock'] + self.eoq]
        self.reorder_point = round(p['safety_stock'] + p['d_mu'] * (self.eoq * p['t_e'] + p['t_r']))
        self.replen_due = 0
        self.replen_open = 0
        
    def avg_level(self):
        """
        provides the average inventory level at any time
        """
        return sum(self.levels[i] for i in range(len(self.levels))) / len(self.levels)

    def level(self):
        """
        provides the average inventory level (the last one in the list) 
        at any time
        """
        return self.levels[len(self.levels) - 1]

    def put(self, amount):
        """
        adds given amount to inventor
        """
        assert amount >= 0, 'Inventory put amount must be >= 0' 
        #self.t.append(self.env.now)
        self.levels.append(self.level() + amount)
        
    def get(self, amount):
        """
        substracts given amount from inventor
        """
        assert self.level() >= amount, 'Inventory get amount must be lower than current level' 
        assert amount >= 0, 'Inventory get amount must be >= 0' 
        #self.t.append(self.env.now)
        self.levels.append(self.level() - amount)


def printfunc_inv(item):
    print("Current level", item.avg_level())
    print("Average level ", item.level())
    print("EOQ ", item.eoq)
    print("Levels", item.levels)
    print("Reorder point ", item.reorder_point)
    print("Replen_due ", item.replen_due)
    print("Replen_open ", item.replen_open)
    print("----------------------------------")

item1 = Inventory(p1)
printfunc_inv(item1)
item1.put(100)
printfunc_inv(item1)
item1.get(50)
printfunc_inv(item1)