"""
This is a rewritten file from scratch for folder environment 
"""

# Importing modules
import matplotlib.pyplot as plt
# ADD OTHER STUFF HERE

# Set the global variables
DAYS_PER_WEEK = 7
WEEKS_PER_YEAR = 52
DAYS_PER_YEAR = WEEKS_PER_YEAR * DAYS_PER_WEEK
HOURS_PER_DAY = 24
SEC_PER_HOUR = 3600
SEC_PER_DAY = SEC_PER_HOUR * HOURS_PER_DAY

# Set the central scaling factor for the model
ONE_DAY = 1.0
FIFTEEN_MINUTES = (ONE_DAY / (HOURS_PER_DAY * 60)) * 15

# Set the start and end of the simulation
SIM_CYCLES = 500
SIM_START = 0
SIM_END = SIM_CYCLES * ONE_DAY

# Type of the distribution to generate random data
DISTRIBUTION_TYPE = "lognorm"
# If it is possible to provide partial shipments or not
ALLOW_PARTIAL_SHIPMENTS = False #True will not affect any code, since the environment is built differently
# Rounding base for generating demands and forecasts
ROUNDING_BASE = 1000                                                    #WHY IS 1000 HERE INSTEAD OF 100?
# Number of days for calculating the forecast
DAYS_FC = 180
# In case of FIXED ORDER PERIOD quantity determination, setting the span in periods
FOP_PERIODS = 30
# Setting the variation coefficient for a machine per production step
MACHINE_VARCOEF = 0.1
# Scheduling regime used for sorting the job list NOT USED IN THE NEW VERSION
SCHEDULING_REGIME = "SPT"
# Number of working hours of a machine in a day (the same as hours per day)
# NOT USED
MACHINE_WORKING_HOURS = HOURS_PER_DAY

"""
Inventory more:
    - 0: No initial amount 
    - 1: Initial amount = safety stock
    - 2: Initial amount = safery stock + static eoq
"""
INVENTORY_MODE = 2
# Maximum level of inventory for both products 
MAXIMUM_INVENTORY = 50_000_000

"""
There are only 2 products in this simulation. 
1st product: niche product, low demand, high volatility
2nd product: volume product, high demand, low volatility
2
The read machine OEE is anywhere between 69& and 82%.
"""

QUANT_MU_1 = 24_400
DIST_DAYS_1 = 1.9
P1 = {
    "d_mu"              : QUANT_MU_1,                                           # number of pieces per day produced
    "d_varcoef"         : 0.75,                                                 # variation coefficient for demand determination
    "d_t_mu"            : DIST_DAYS_1,                                          # number of days on average between 2 orders
    "d_fc_noise"        : 1.0,                                                  # variation coefficient for forecast determination 0 to 1 as percentage on top of it
    "safety_stock"      : round((QUANT_MU_1 / DIST_DAYS_1) * DAYS_PER_WEEK),    # safety stock for one week
    "E_p"               : 0.03,                                                 # cost per piece for EOQ calculation
    "B_k"               : 200,                                                  # setup cost per batch for EOQ calculation  
    "irate"             : 0.1,                                                  # interest rate for inventory for EOQ calculation
    "t_e"               : 0.22 / SEC_PER_DAY,                                   # time to produce one piece in fraction of a day      
    "t_r"               : 2.0 / HOURS_PER_DAY,                                  # time to setup a batch in fraction of a day            
    "eoq_mode"          : "FOW"                                                 # mode for EOQ calculation (check below the definitions)
}

QUANT_MU_2 = 978_000
DIST_DAYS_2 = 1.58
P2 = {
    "d_mu"              : QUANT_MU_2,                                           # number of pieces per day produced
    "d_varcoef"         : 0.75,                                                 # variation coefficient for demand determination
    "d_t_mu"            : DIST_DAYS_2,                                          # number of days on average between 2 orders
    "d_fc_noise"        : 1.0,                                                  # variation coefficient for forecast determination 0 to 1 as percentage on top of it
    "safety_stock"      : round((QUANT_MU_2 / DIST_DAYS_2) * DAYS_PER_WEEK),    # safety stock for one week
    "E_p"               : 0.03,                                                 # cost per piece for EOQ calculation
    "B_k"               : 200,                                                  # setup cost per batch for EOQ calculation  
    "irate"             : 0.1,                                                  # interest rate for inventory for EOQ calculation
    "t_e"               : 0.22 / SEC_PER_DAY,                                   # time to produce one piece in fraction of a day      
    "t_r"               : 2.0 / HOURS_PER_DAY,                                  # time to setup a batch in fraction of a day            
    "eoq_mode"          : "FOW"                                                 # mode for EOQ calculation (check below the definitions)
}

# ADD HERE SOME FUNCTIONS