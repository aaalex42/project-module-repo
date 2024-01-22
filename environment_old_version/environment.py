# -*- coding: utf-8 -*-
"""
Simulates a plant with one machine and two products
Implements:
    - demand and forecast generation
    - demand fulfillment from stock
    - stock reorder point observation and control
    - dynamic economic order quantity calculation
    - sequence scheduling

Limitations:
- The time unit is 1 for one day. First day is 0.
- Rough and ready, some of the patterns have not been tested yet
- Currently limited to 1 order per product per day, bucket is also 1 day
- no machine breakdowns implemented yet

@author: ruekgauer

Changelog:
    V 1.0  17.4.23 Initial external version
           24.4.23 minor adjustment: OEE is w/o setup time
    V 1.1   2.5.23 Inventory init uses the very regime later required for sim
                   at startup, an expected OEE is calculated
    V 2.0  16.5.23 Demand determination changed to compounded Poisson
                   process which better reflects reality. This requires major
                   adjustments, therefore major revision change
                   added demand and fulfillment tracking
    V 2.1  19.6.23 added validated real demand data (former data error eliminated)
           22.6.23 corrected forecast calculation offset error by introducing carryover
                   minor fixes here and there
                   corrected service level calc, skip d=0 incidences
                   minor pythonizations all over the place
                   dynamized main, added comments
    V 2.2  26.6.23 adjusted demand data according to new information from client
            6.7.23 Added some statistics: inv.n_put/n_get, dmnd.n_jobs
    V 2.3   7.7.23 Fixed round and trunc errors in demand and forecast generation
                   introduced functions for both for easier debugging
                   OEE looks much better now, also number of prod jobs
            8.7.23 Added "." for sim progress with progress_indicator process
    V 2.4 21.11.23 Added four_weeks / FOW eoq model
                   Minor bug fix in gen_fc and Inv init routine
                   some cosmestic adjustments
"""

"""
Importing the libraries
"""
import matplotlib.pyplot as plt
from base_functions import *
from classes import *


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
SIM_CYCLES = 500
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
RBASE = 1000
# size of the forecast for all products:
DAYS_FC = 180
# span in periods used in case of fixed order period quantity determination:
FOP_PERIODS = 30
# var coef for machine execution per propdution step:
MACHINE_VARCOEF = 0.1
# scheduling regime used herein, see defitions below:
SCHEDULING_REGIME = 'SPT'

"""
We are dealing with 2 products herein, their corresponding economic framework 
defined as follows

The demand is double stochastic: a Poisson time distance, and a (not quite) Lognorm amount variation

1. Niche product, low demand, high volatility
2. Volume product: high demand, low volatility
        
The real machine OEE is anywhere between 69% and 82%
"""
QUANT_MU = 24400 # product 1 is low amount, high volatility, high uncertainty
DIST_DAYS = 1.9 
p1 = { \
    'd_mu'         : QUANT_MU, # number of pieces per day produced \
    'd_varcoef'    : 0.75, # var coef for demand determination \
    'd_t_mu'       : DIST_DAYS, # number of days on avg between 2 orders \
    'd_fc_noise'   : 1.0, # var coef of forecast noise 0...1 as percentage of on top of it \
    'safety_stock' : 1.0 * QUANT_MU / DIST_DAYS * DAYS_PER_WEEK, # safety stock: 1 week \
    'E_p'          : 0.03,  # cost per piece for economic order quantity calc \
    'B_k'          : 200,  # setup cost per batch for eoq calc \
    'irate'        : 0.1,  # inventory interest for eoq calc \
    't_e'          : 0.22 / SEC_PER_DAY,  # production time per piece in fractions of a day \
    't_r'          : 2.0 / HOURS_PER_DAY,  # setup time per batch in fractions of a day \
    'eoq_mode'     : 'FOW' # which eoq model to use, definitions see below \
}

QUANT_MU = 978000 # product 2 is high amount, low volatility, medium uncertainty
DIST_DAYS = 1.58
p2 = { \
    'd_mu'         : QUANT_MU, \
    'd_varcoef'    : 0.75,  \
    'd_t_mu'       : DIST_DAYS, \
    'd_fc_noise'   : 0.5, \
    'safety_stock' : 1.0 * QUANT_MU / DIST_DAYS * DAYS_PER_WEEK, \
    'E_p'          : 0.04, \
    'B_k'          : 300,  \
    'irate'        : 0.1,  \
    't_e'          : 0.11 / SEC_PER_DAY, \
    't_r'          : 2.0 / HOURS_PER_DAY, \
    'eoq_mode'     : 'FOW' \
}
 
master_product_data = (p1, p2)

        
"""
from here on the process wrappers
"""

def produce(env, pp):
    """ 
    simple process wrapper around production plan execution part and inventory update 
    runs a production plan, so it runs one machine
    produce has no bucket, as it continuesly processes all jobs pending
    """
    while True:
        dur, amount, inv = pp.produce()
        if dur > 0:
            # wait duration - note if breakdowns are implmented, the waiting time can be longer
            yield env.timeout(dur)
            
            # store product after the production is finished, no earlier!
            inv.put(amount)
            inv.replen_due -= amount
        else:
            # wait a bit to prevent infinite loop
            yield env.timeout(FIFTEEN_MINUTES)    

def fulfill(env, d, inv, allow_part_shipments, bucket):
    """
    demand fulfillment process for one set of demand and inventory, per bucket
    env : simpy environment
    d : demand instance
    inv : inventory process instance
    """
    while True:
        # provide all material to demand, deduct consumed amount from inventory
        max_mat = inv.level()
        mat_used = d.fulfill(max_mat, allow_part_shipments) 
        inv.get(mat_used)
        yield env.timeout(bucket)


def plan(env, pp, bucket):
    """
    run one planning cycle per bucket
    so plan for 1 machine
    """
    while True:
        pp.plan(env.now)
        yield env.timeout(bucket)

def progress_indicator(env, dur):
    """
    show that the simulation is still running ...
    """
    while True:
        print('.', end = '')
        yield env.timeout(dur)

"""
finally the actual program
"""

import simpy
import datetime

if __name__ == '__main__':

    """
    22.6.23: dynamic version with as many products on one machine as given 
             my the master product data list
    """
    # init random engine
    random.seed(datetime.datetime.now().second)
    # init sim environment
    env = simpy.Environment(initial_time = SIM_START)

    # init demands, inventories per product, and production plan and execution per machine
    dl = (Demand(p1, RBASE, DAYS_FC, env), 
          Demand(p2, RBASE, DAYS_FC, env))
    ##for p in master_product_data:
    ##   dl.append(Demand(p, RBASE, DAYS_FC, env))

    avg_OEE = (dl[0].avg_daily_machine_time() + dl[1].avg_daily_machine_time()) / ONE_DAY
    ##for d in dl:
    ##    avg_OEE += d.avg_daily_machine_time() / ONE_DAY
    print(f'Expected OEE = {avg_OEE}')

    il = (Inventory(dl[0], env), 
          Inventory(dl[1], env))
    ##for d in dl:
    ##   il.append(Inventory(d, env))
    
    demand_and_inv_pipeline = (
        {'d': dl[0], 'inv': il[0]}, 
        {'d': dl[1], 'inv': il[1]}
    )
    ##demand_and_inv_pipeline = [{'d': d, 'inv': i} for d, i in zip(dl, il)]
    pp = ProdPlan(SCHEDULING_REGIME, MACHINE_VARCOEF, demand_and_inv_pipeline)

    # launch fulfillments per product, and planning and production per machine
    for d, i in zip(dl, il):
        env.process(fulfill(env, d, i, ALLOW_PART_SHIPMENTS, ONE_DAY))
    env.process(plan(env, pp, ONE_DAY))
    env.process(produce(env, pp))
    
    # show that sim is still running
    env.process(progress_indicator(env, ONE_DAY))
    
    # run sim
    env.run(until = SIM_END)
    
    # post production: data analysis if desired
    if True:

        for i, (inv, p, di) in enumerate(zip(il, master_product_data, dl)):

            # plot inventory levels over time
            plt.figure(figsize=(8,8))
            plt.plot(inv.t, inv.levels) # , label='inventory ' + str(i+1))
            plt.legend(loc = 'upper left')
            plt.title(f'Inventory {i+1}')
            plt.xlabel('sim cycle = days')
            plt.ylabel('amount')
            plt.rcParams.update({'font.size': 8})
            plt.show()
            print(f'product {i+1}')
            # show actual avg inventory levels in pcs
            print(f'Actual avg inventory {i+1} level = {round(inv.avg_level())}')

            # show actual avg inventory levels in weeks
            avg_dmnd_per_week = p['d_mu'] * DAYS_PER_WEEK / p['d_t_mu']
            avg_inv_weeks = round(inv.avg_level() / avg_dmnd_per_week * 10.0) / 10.0
            print(f'Actual avg inventory {i+1} level = {avg_inv_weeks} weeks')
            
            print(f'number of demand events = {di.n_good + di.n_bad}')
            print(f'number of production jobs = {di.n_jobs}')
            print(f'number of inbound storage events = {inv.n_put}')
            print(f'number of outbound storage events = {inv.n_get}')

        for i, d in enumerate(dl):

            # show actual service level         
            print(f'Actual service level demand {i+1} = {d.service_level()}')

            # plot demand, fulfillment, backlog
            plt.figure(figsize=(8,8)) 
            plt.plot(d.t, d.d, label='demand')
            plt.plot(d.t, d.ff, label='fulfilled')
            plt.plot(d.t, d.delta_bl, label='delta backlog')
            plt.legend(loc = 'upper left')
            plt.title(f'Demand {i+1}')
            plt.xlabel('sim cyle = days')
            plt.ylabel('amount')
            plt.rcParams.update({'font.size': 8})
            plt.show()

        # show actual OEE:
        OEE = pp.prod_time / (SIM_END - SIM_START)
        print(f'Actual OEE = {OEE}')