from base_functions import *
from environment import DAYS_PER_YEAR, DAYS_PER_WEEK, FOP_PERIODS


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
    6-20-23:  fix for FOP: from d[0:x-1] to d[0:x]
    11-21-23: added FOW for four weeks of material, it is however almost 
              identical to FOP = 28
    """
    if mode == 'Andler':
        # Andler, note this is not correct as the equation assumes constant demand
        sum_d = sum(d)
        len_d = len(d)
        J = round(sum_d / len_d * DAYS_PER_YEAR)
        return eoq_static(J, E_p, B_k, Z)
    elif mode == 'FOP':
        # fixed order period
        return sum(d[0:min(FOP_PERIODS, len(d))])
    elif mode == 'FOW':
        # four week method utilized by ITW: add 4 weeks of stock
        # calculate avg daily demand
        avg_daily_dmnd = sum(d) / len(d)
        # return 4 weeks of demand
        return avg_daily_dmnd * DAYS_PER_WEEK * 4
    else:
        raise ValueError('{mode} not defined')


class Demand:
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
    
    7-6-23: added n_jobs and related functionality
    """
    def __init__(self, product, rbase, d_fc, env):
        """
        inits everything and generates an initial demand pattern stream
        p: product
        rbase: rounding base
        d_fc: days forecast
        """
        self.product = product      # remembers product
        self.rbase = rbase    # rounding base
        self.n_good = 0       # counts the successful demand fulfillments
        self.n_bad = 0        # counts the unsuccessful ones (not on time or amount)
        self.backlog = 0      # stores the unfulfilled demands for later
        self.demands = []     # holds the actual demands[d_fc], subject, is a ring buffer with one entry for every day
        self.forecasts = []   # holds the noised up forecasts [d_fc, first entry is tomorrow] with one entry for every day
        self.nextd = 0        # next event in days where demand is generated
        self.curtd = 0        # pointer to current time relative to nextd
        self.t = []           # time stamp for fulfillment, just for analysis 
        self.ff = []          # fulfilled amount
        self.d = []           # past actual demanded recorded for debugging und display purposes only
        self.delta_bl = []    # delta backlog
        self.env = env        # dto, just for analysis puposes
        self.n_jobs = 0       # number of production jobs scheduled over time

        # initially build demand stream, 1st is current demand
        self.nextd = poisson_int(product['d_t_mu'], 1) 
        self.gen_demand_and_fc(d_fc + 1) 
    
    def gen_demand_and_fc(self, amount):
        """
        generates new demand, simply a cast
        important: appends at the end (fulfill will purge at the front)
        and updates fc for demands other than current (1st in list)
        6-22-23: forecast corrected, was artificially high due to left = 0 trunc, 
                 introduced carryover
        7-6-23: added function wrappers due to bug fix, made debugging easier
        """
        

        self.curtd, self.nextd, new_demands = gen_demands(self.product['d_mu'], self.product['d_varcoef'], \
                                                          self.product['d_t_mu'], self.rbase, \
                                                          self.nextd, self.curtd, amount)

        self.demands = self.demands + new_demands

        self.forecasts = gen_fc(self.demands[1:], self.product['d_fc_noise'], \
                                self.product['d_mu'], self.product['d_varcoef'], self.rbase)
        
            
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
        22.6.23: fixec step 2: only if demand > 0,otherwise the n_good counter is not correct
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
        # 2. then handle current demand step if necessary as much as possible and update statistics
        if self.demands[0] > 0:
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
        daily_demand = self.product['d_mu'] / self.product['d_t_mu']
        daily_prod_time = daily_demand * self.product['t_e']
        eoq = eoq_dynamic(self.demands, self.product['E_p'], self.product['B_k'], self.product['irate'], self.product['eoq_mode'])
        daily_setup_time = self.product['d_mu'] / self.product['d_t_mu'] / eoq * self.product['t_r'] 
        return daily_prod_time + daily_setup_time


class Inventory:
    """ 
    - passive device just holding inventory level and reorder point info 
      initial amount = safety + static eoq
    - reorder point also static for replenishment cycle based on eoq * t_e + t_r
    - also stores all past levels for later analysis both in time and amounts
    - this class also holds the information for replenishment handling: at the
      beginning of a replen cycle, replen_due and replen_open are set to the least 
      amount = rop - level, then this routine counts down replen_due and the
      fulfillment routine counts down replen_open
      
    Invenory is always for one product
    
    from V1.1 it uses the same quantity regime as planning uses
    
    7-6-23: added n_put and n_get and related functionality
    11-21-23: added some comments to describe what we actually do
              minor bug fix on ROP
    """
    def __init__(self, d, env):
        """
        d: demand instance
        env: simpy environment, only needed for env.now (statisical purposes)
        """

        product = d.product
        self.env = env
        self.t = [self.env.now]

        # calculate economic quantity based on current conditions
        eoq = eoq_dynamic([d.demands[0]] + d.forecasts,
                          product['E_p'], 
                          product['B_k'], 
                          product['irate'], 
                          product['eoq_mode'])
        
        # set starting inventory at full: safety stock+ economic quantity
        self.levels = [product['safety_stock'] + eoq]

        # set reorder point at safety stock + consumption rate x replenishment time
        # with replenishment time per piece = setup time / current economic quanity x+ production time 
        replen_time_per_cycle = eoq * product['t_e'] + product['t_r']
        consumption_rate = product['d_mu'] / product['d_t_mu']
        consump_per_replen_cycle = consumption_rate * replen_time_per_cycle 
        self.reorder_point = round(product['safety_stock'] + consump_per_replen_cycle)
        
        self.replen_due = 0
        self.replen_open = 0
        self.n_put = 0
        self.n_get = 0
        
    def avg_level(self):
        """
        provides the average inventory level at any time
        """
        return sum(self.levels[i] for i in range(len(self.levels))) / len(self.levels)

    def level(self):
        """
        provides the average inventory level (the last one in the list) 
        at any time
        11-21-23 pytonized [-1]
        """
        return self.levels[-1]

    def put(self, amount):
        """
        adds given amount to inventor
        """
        assert amount >= 0, 'Inventory put amount must be >= 0' 
        self.t.append(self.env.now)
        self.levels.append(self.level() + amount)
        self.n_put += 1
        
    def get(self, amount):
        """
        substracts given amount from inventor
        """
        assert self.level() >= amount, 'Inventory get amount must be lower than current level' 
        assert amount >= 0, 'Inventory get amount must be >= 0' 
        if amount == 0:
            return
        self.t.append(self.env.now)
        self.levels.append(self.level() - amount)
        self.n_get += 1


class ProdPlan:
    """
    class for production plan and execution for any one machine
    ProdPlan links one machine with a queue of product demands and inventories
    so the architecture is as follows:
        machine --> list of {demands, inventories always going together}
    implements
    - planing step including reorder point, economic quantity and order scheduling
    - actual scheduling
    - production step
    """
    def __init__(self, scheduling_regime, varcoef, di_queue):
        """
        scheduling regime: see below
        varcoef: sigma/mu for any one production stepsize
        di_queue: pipeline of demands and inventories
        internal: job list holding the production pipeline
        """
        self.joblist = []
        self.scheduling_regime = scheduling_regime
        self.last_prod = None # remember past product produced to add setup time
        self.varcoef = varcoef # variance coef for both setup and execution
        self.dem_inv_queue = di_queue
        self.prod_time = 0 # stores the pure execution time
        self.setup_time = 0 # stores the pure setup time
        
    def plan(self, now):
        """ 
        this routine adds prod jobs, the produce routine eliminates them again 
        starts with adding new orders (generating new demand)
        then takes demand/inventory pipeline, fills job pipeline
        """
        # for all demand and inventory tuples
        for dem_inv in self.dem_inv_queue:

            dem = dem_inv['d']
            inv = dem_inv['inv']
    
            # 1. get new demand and forecast
            dem.gen_demand_and_fc(1) 

            # 2. check reorder point
            # . check and start new cycle if needed
            if inv.level() <= inv.reorder_point and inv.replen_due <= 0:

                inv.replen_due = inv.reorder_point - inv.level()
                inv.replen_open = inv.replen_due

            # . keep adding jobs if not yet enough issued 
            if inv.replen_open > 0:

                # 3. economic quantity
                eoq = eoq_dynamic([dem.demands[0]] + dem.forecasts,
                                  dem.product['E_p'], 
                                  dem.product['B_k'], 
                                  dem.product['irate'], 
                                  dem.product['eoq_mode'])
                
                # yes, it is possible that we are below rop and yet eoq = 0!
                if eoq > 0: 

                    # add to job list
                    self.joblist.append({'amount': eoq, 'date': now, 'inv': inv, 'p': dem.product})
                    inv.replen_open -= eoq
                    dem.n_jobs += 1

        # 4. sequence scheduling
        self.schedule()
 
    def schedule(self):
        """
        several classic job sequence scheduling algorithms are provided
        so far, only SPT has been used and tested!
        a sorter column is added, used and removed again
        22.6.23: pythonized new version
        """

        # generate sorter column depending on scheduling regime
        if self.scheduling_regime == 'Value':

            ascend = False
            self.joblist = [jb | {'sorter': jb['amount'] * jb['p']['E_p']} for jb in self.joblist ]

        elif self.scheduling_regime == 'SPT':

            ascend = True
            self.joblist = [jb | {'sorter': jb['amount'] * jb['p']['t_e']} for jb in self.joblist ]

        elif self.scheduling_regime == 'LPT':

            ascend = False
            self.joblist = [jb | {'sorter': jb['amount'] * jb['p']['t_e']} for jb in self.joblist ]

        elif self.scheduling_regime == 'Slip':

            ascend = True
            self.joblist = [jb | {'sorter': jb['date'] - jb['amount'] * jb['p']['t_e']} for jb in self.joblist ]

        else:

            raise ValueError('{scheduling_regime} not defined') 
        
        # sort by sorter
        self.joblist = sorted(self.joblist, key = lambda k: k['sorter'], reverse = ascend)

        # eliminate sorter
        self.joblist = list(filter(lambda x: x.pop('sorter', None) or True, self.joblist))


    def produce(self):
        """ 
        execute first job in the list and eliminate entry 
        deliverables are:
        - duration of production job, 
        - amount of parts produced, 
        - inventory to store the parts into
        storage is handled outside, because we have to wait for the production to be finished
        first. this is handled outside to separate functions
        """

        if len(self.joblist) == 0:
            return 0, 0, 0
        
        job = self.joblist[0]
        assert job['amount'] > 0, 'production amount must be > 0'

        # add production time to duration
        # note varcoef applies to every prod step, so due to addition of variances we must divide by sqrt(amount)
        duration = distri(job['amount'] * job['p']['t_e'], self.varcoef / numpy.sqrt(job['amount']))
        self.prod_time += duration

        # add setup time to duration, but only if product switch. remember current product produced
        setup_time = distri(job['p']['t_r'], self.varcoef) if self.last_prod != job['p'] else 0.0
        duration += setup_time
        self.setup_time += setup_time
        self.last_prod = job['p']

        # remove the finished job from list
        del self.joblist[0]

        return duration, job['amount'], job['inv']