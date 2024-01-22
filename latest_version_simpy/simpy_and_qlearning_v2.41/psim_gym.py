# -*- coding: utf-8 -*-
"""
Based on 2.41
just one product so far, this is inconsequencial, see below
gym wrapper
11.12.23
"""
import psim_1mach_2prod_v241 as sim

SIM_CYCLES = 2000
SIM_START = 0.0
SIM_END = SIM_CYCLES * sim.ONE_DAY

VERBOSE = False

"""
AR 11.12.23 change: increase utilization for one product a bit
"""
p3 = sim.p2
p3['d_mu'] *= 1.3
 
this_master_product_data = [p3]


class ProdPlanAI(sim.ProdPlan):
    """
    special version of ProdPlan for agent-driven operation
    """
    def __init__(self, scheduling_regime, varcoef, di_queue):
        super(ProdPlanAI, self).__init__(scheduling_regime, varcoef, di_queue)
        
    def plan(self, now, prod_volume):
        """
        generates new demand
        returns current new demand as list with one entry for every product
        adds production job to list for all products, if prod_volume > 0
        (this list will be purged completely at the end of the env step)
        so this is a vector in (prod volume per product) / vector out (new demand per product) 
        function
        """
        new_demand = []
        # for all demand and inventory tuples
        for di, pvi in zip(self.di_queue, prod_volume):
            d = di['d']
            inv = di['inv']
            p = d.p
            # 1. get new demand and forecast
            d.gen_demand_and_fc(1) 
            # current demand:
            new_demand.append(d.demands[0])
            # add job to list if amount is > 0
            if pvi > 0:
                self.joblist.append({'amount': pvi, 'date': now, 'inv': inv, 'p': p})
            d.n_jobs += 1
        return new_demand
 
    def schedule(self):
        """ not used in this class """
        pass
    
    def purge_joblist(self):
        """ 
        cleans up the joblist at the end of an env.step to start up fresh 
        again in the next step 
        """
        self.joblist = []

def produceAI(env, pp):
    """ 
    difference to the original version: no replen_due mgt
    """
    while True:
        dur, amount, inv = pp.produce()
        if dur > 0:
            # wait duration - note if breakdowns are implmented, the waiting time can be longer
            yield env.timeout(dur)
            
            # store product after the production is finished, no earlier!
            inv.put(amount)
        else:
            # wait a bit to prevent infinite loop
            yield env.timeout(sim.FIFTEEN_MINUTES)    


    
""" 
from here on we are for now dealing with one product only. this is inconsequencial as abote
everything was vectorized. we can change it below later

also the handing of bins is not treated well in the agent. therefore we do it herein. we limit
it to 100 bins each for inventory, demand, and production


static properties used outside 
------------------------------
max_inventory
max_demand
max_action
action_space

dynamic properties used outside
-------------------------------
demand_open
new_demand

actions used outside
--------------------
init()
obs = [inventory, demand_open], reward, done, _ = step(action = next production volume expected)
obs = reset()
"""        

import simpy
import datetime

import gymnasium
import numpy as np
from gymnasium import spaces
import random


class InventoryManagementEnv(gymnasium.Env):

    def __init__(self):

        super(InventoryManagementEnv, self).__init__()

        self.demand_open = 0
        self.new_demand = 0
                
        # init random engine
        random.seed(datetime.datetime.now().second)

        # init sim environment
        self.env = simpy.Environment(initial_time = SIM_START)

        # init demands
        self.dl = []
        for p in this_master_product_data:
            self.dl.append(sim.Demand(p, sim.RBASE, sim.DAYS_FC, self.env))
    
        avg_OEE = 0
        for d in self.dl:
            avg_OEE += d.avg_daily_machine_time() / sim.ONE_DAY
        print(f'Expected OEE = {avg_OEE}')
        
        # set max demand to x the current max demand in the list
        import numpy as np
        NUM_DEMAND_BINS = 100
        MAX_DEMAND_FACTOR = 5
        demands = self.dl[0].demands ##### currently only one product!!!!!
        max_abs_demand = MAX_DEMAND_FACTOR * max(demands)
        self.max_demand = NUM_DEMAND_BINS
        self.demand_bins = np.linspace(0, max_abs_demand, self.max_demand, dtype=np.int32)
        
        # set max inventory to 25 weeks avg demand which is twice the normal
        NUM_INV_BINS = 100
        MAX_NUM_INV_WEEKS = 25
        p = this_master_product_data[0]  ##### currently only one product!!!!!
        daily_avg_demand = p['d_mu'] / p['d_t_mu']
        max_abs_inventory = MAX_NUM_INV_WEEKS * sim.DAYS_PER_WEEK * daily_avg_demand
        self.max_inventory = NUM_INV_BINS
        self.inventory_bins = np.linspace(0, max_abs_inventory, self.max_inventory, dtype=np.int32)
        
        # set max prod volume to 5x the current production volume
        MAX_NUM_PROD_EOQS = 5
        NUM_PROD_BINS = 100
        self.max_action = NUM_PROD_BINS
        self.action_space = spaces.Discrete(self.max_action) # what is this really needed for?
        eoq = sim.eoq_dynamic(demands, p['E_p'], p['B_k'], p['irate'], p['eoq_mode'])
        max_prodvol = MAX_NUM_PROD_EOQS * eoq
        self.prodvol_bins = np.linspace(0, max_prodvol, self.max_action, dtype=np.int32)
    
        # init inventories
        self.il = []
        for d in self.dl:
            self.il.append(sim.Inventory(d, self.env))

        # init inventories and demands pipeline and instanciation of prod plan            
        self.demand_and_inv_pipeline = [{'d': d, 'inv': i} for d, i in zip(self.dl, self.il)]
        self.pp = ProdPlanAI(sim.SCHEDULING_REGIME, sim.MACHINE_VARCOEF, self.demand_and_inv_pipeline)

    def step(self, action):

        # obtain new demand and update env to action
        prodvol = self.prodvol_bins[action]
        new_demands = self.pp.plan(self.env.now, [prodvol])
        self.new_demand = np.digitize(new_demands[0], self.demand_bins) - 1
        # new demands are never used and not passed on to the agent?!!=!=!

        # fulfill
        for d, i in zip(self.dl, self.il):
            # provide all material to demand, deduct consumed amount from inventory
            max_mat = i.level()
            mat_used, fulfilled = d.fulfill(max_mat, sim.ALLOW_PART_SHIPMENTS) 
            i.get(mat_used)

        # continue sim for one step
        self.env.run(until =  min(self.env.now + sim.ONE_DAY, SIM_END))

        # Check if the episode is done
        if self.env.now >= SIM_END:
            done = True
        else:
            done = False
        
        # Positive reward for meeting or exceeding the service level target/no backlog
        reward = 1  
        # subtract 2 if the prod jobs from the last round could not be implemented on the machine
        joblist_reward = -2 if len(self.pp.joblist) > 0 else 0
        reward += joblist_reward 
        # subtract up to 5 for inventory
        inv = self.il[0].levels[-1]
        inv_bin = np.digitize(inv, self.inventory_bins) - 1
        inv_reward = -inv_bin / len(self.inventory_bins) * 5
        reward += inv_reward
        # penalize service level missed with -10
        sl_reward = -10 if not fulfilled else 0
        reward += sl_reward
        # alternative, high cost ...
        #MIN_SERVICE_LEVEL = 0.95
        #reward -= 10 if self.dl[0].service_level() < MIN_SERVICE_LEVEL else 0
        if VERBOSE:
            print(f'reward = {reward} with jl = {joblist_reward}, inv = {inv_reward}, sl = {sl_reward}')

        # clean up the job list
        self.pp.purge_joblist()

        # backlog bin, match with inventory bins as this can be large!
        backlog_bin = np.digitize(self.dl[0].backlog, self.inventory_bins) - 1

        return np.array([inv_bin, backlog_bin]), reward, done, {}

    def reset(self):
        """
        resets and starts sim
        """
        self.demand_open = 0
        self.new_demand = 0
                
        # init random engine
        random.seed(datetime.datetime.now().second)

        # re-init sim environment
        self.env = simpy.Environment(initial_time = SIM_START)

        # re-init demands
        self.dl = []
        for p in this_master_product_data:
            self.dl.append(sim.Demand(p, sim.RBASE, sim.DAYS_FC, self.env))

        # re-init inventories
        self.il = []
        for d in self.dl:
            self.il.append(sim.Inventory(d, self.env))

        # re-init inventories and demands pipeline and instanciation of prod plan            
        self.demand_and_inv_pipeline = [{'d': d, 'inv': i} for d, i in zip(self.dl, self.il)]
        self.pp = ProdPlanAI(sim.SCHEDULING_REGIME, sim.MACHINE_VARCOEF, self.demand_and_inv_pipeline)
    
        # launch production 
        self.env.process(produceAI(self.env, self.pp))

        # get initial demand
        self.demand_open = 0
        demands = self.dl[0].demands
        self.new_demand = np.digitize(demands[0], self.demand_bins) - 1

        # fulfill
        for d, i in zip(self.dl, self.il):
            # provide all material to demand, deduct consumed amount from inventory
            max_mat = i.level()
            mat_used, _ = d.fulfill(max_mat, sim.ALLOW_PART_SHIPMENTS) 
            i.get(mat_used)

        # run sim for one step
        self.env.run(until =  min(self.env.now + sim.ONE_DAY, SIM_END))

        inv = self.il[0].levels[-1]
        inv_bin = np.digitize(inv, self.inventory_bins) - 1
        backlog_bin = np.digitize(self.dl[0].backlog, self.inventory_bins) - 1

        return np.array([inv_bin, backlog_bin])
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def post_sim_analysis(self):
        
       import matplotlib.pyplot as plt

       for i, (inv, p, di) in enumerate(zip(self.il, this_master_product_data, self.dl)):

           # plot inventory levels over time
           plt.figure(figsize=(8,8))
           plt.plot(inv.t, inv.levels) 
           plt.legend(loc = 'upper left')
           plt.title(f'Inventory {i+1}')
           plt.xlabel('sim cyle = days')
           plt.ylabel('amount')
           plt.rcParams.update({'font.size': 8})
           plt.show()
           print(f'product {i+1}')
           # show actual avg inventory levels in pcs
           print(f'Actual avg inventory {i+1} level = {round(inv.avg_level())}')

           # show actual avg inventory levels in weeks
           avg_dmnd_per_week = p['d_mu'] * sim.DAYS_PER_WEEK / p['d_t_mu']
           avg_inv_weeks = round(inv.avg_level() / avg_dmnd_per_week * 10.0) / 10.0
           print(f'Actual avg inventory {i+1} level = {avg_inv_weeks} weeks')
           
           print(f'number of demand events = {di.n_good + di.n_bad}')
           print(f'number of production jobs = {di.n_jobs}')
           print(f'number of inbound storage events = {inv.n_put}')
           print(f'number of outbound storage events = {inv.n_get}')

       for i, d in enumerate(self.dl):

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
       OEE = self.pp.prod_time / (SIM_END - SIM_START)
       print(f'Actual OEE = {OEE}')    
    
