"""
TODO:
    - class Demand function fulfill: think how to implement time step t smarter.
    - add a simple noise instead of lognormal for setup and run time. It will be the same (in terms of idea)
        and faster
    
"""

from base_functions_n import *
from main import *


class Demand:
    """
    This class implements:
        - demand generation
        - fulfillment handling
        - backlog handling
        - statistical fulfillment analysis
        - (LEFT OUT: forecast generation)
    
    One demand per day is assumed for a product.
    For example, two separate demand classes for 2 products.
    Whenever a demand is requested, it will be processed (and new demand per day is "generated")
    
    Changes:
        - 11.12.23: RBASE and DAYS_FC are used as global variables
        - 11.12.23: gen_demand_and_fc function is left out. Demand is generated in the __init__
        - 11.12.23: past demands list is not necessary, since it is the same as the demands list
        - 11.12.23: fulfill function: ALLOW_PARTIAL_SHIPMENTS as global variable

    Comments from previous version:
        - from V1.1 it is necessary to generate the full demand list as it is used for Inventory class
        - 06.07.23 added n_jobs and related functionality 
    """

    def __init__(self, product):
        self.product = product            # product description
        self.n_good = 0                   # number of fulfilled orders
        self.n_bad = 0                    # number of not fulfilled orders
        self.backlog = 0                  # backlog is the amount of demand that is not fulfilled
        self.demand = gen_demands(        # generate the demand here with the vectorized function
            self.product["d_mu"],
            self.product["d_varcoef"],
            self.product["d_t_mu"],
            SIM_CYCLES
        )
        self.fulfilled_amount = []        # list of fulfilled amounts
        self.backlog_delta = []           # list of differences in backlog
        # ADD HERE MORE

        # Calculate the EOQ
        self.eoq = eoq_dynamic(self.demand,
                          self.product["E_p"],
                          self.product["B_k"], 
                          self.product["irate"],
                          self.product["eoq_mode"])
    

    @property
    def service_level(self):
        """
        This function calculates the service level with the formula:
            No. good shipments (full amount and in time) / No. of all shipments
        """

        return self.n_good / (self.n_good + self.n_bad)
    

    @property
    def avg_daily_machine_time(self):
        """
        This function calculates the average daily machine time per product.

        Changes:
            - 11.12.23: EOQ is calculated in the __init__ function
            - UNKNOWN DATE: from V2.0: daily demand = d_mu / d_t_mu

        Possible problems:
            - too large list of demands (100k or more)
        """
        
        # Calculate the average daily demand
        avg_daily_demand = self.product["d_mu"] / self.product["d_t_mu"]
        # Calculate the average daily production time
        daily_production_time = avg_daily_demand * self.product["t_e"]
        # Calculate the daily setup time
        daily_setup_time = (avg_daily_demand / self.eoq) * self.product["t_r"]

        return daily_production_time + daily_setup_time


    def fulfill(self, material, t):
        """
        This function returns the material consumed at a time step t. 
        Either it fulfills the demand fully or partially (also building backlog)
        If there is previous backlog, it will be fulfilled first.

        Changes:
            - 27.12.23: added exit codes for the RL agent
            - 11.12.23: ALLOW_PARTIAL_SHIPMENTS as global variable
            - 11.12.23: Added a time step t
            - 22.06.23: fixed step 2 - only if demand > 0 n_good is increased
        """

        # set a variable to store if the fulfillment was successful
        exit_code = 10
        # set a variable to track the material consumed
        material_used = 0
        # store the initial backlog
        backlog_init = self.backlog

        # 1. Fulfill the backlog first
        if self.backlog > 0:
            # Check if there is enough material to handle the backlog:
            if material >= self.backlog:
                # Increase the material used by the whole backlog
                material_used += self.backlog
                # Decrease the material by the whole backlog
                material -= self.backlog
                # Set the backlog to 0  
                self.backlog = 0
            
            else:
                # Increase the material used by the whole material
                material_used += material
                # Decrease the backlog by the whole material
                self.backlog -= material
                # Set the material to 0
                material = 0
        
        # 2. If there is some material left, fulfill the current demand (if there is any)
        if material > 0 and self.demand[t] > 0:
            if material >= self.demand[t]:
                # Increase the material used by the whole demand
                material_used += self.demand[t]
                # Increase the number of fulfilled orders
                self.n_good += 1
                # Exit code 11: the demand was fulfilled
                exit_code = 11
            
            else:
                # Exit code 12: the demand was not fulfilled
                exit_code = 12
                # Increase the number of not fulfilled orders
                self.n_bad += 1
                # Check if partial shipments are allowed
                if ALLOW_PARTIAL_SHIPMENTS:
                    # Increase the backjlog by the remaining demand
                    self.backlog += self.demand[t] - material
                    # Increase the material used by remaining material
                    material_used += material
                    # Set the material to 0
                    material = 0
                else:
                    # Increase the backlog by the whole demand
                    self.backlog += self.demand[t]
        
        # Save some statistics
        self.fulfilled_amount.append(material_used)
        # Save the difference in backlog. When positive, it means that the backlog decreased
        # and when negative, it means that the backlog increased
        self.backlog_delta.append(backlog_init - self.backlog) 

        return material_used, exit_code
    

class Inventory:
    """
    This class implements the inventory for one product

    Possible Options for INVENTORY_MODE:
        - 0: No initial amount 
        - 1: Initial amount = safety stock
        - 2: Initial amount = safety stock + static eoq
    
    Changes:
        - 03.01.24: changed the type of inventory list to integer
        - 11.12.23: No reorder point implemented, as this task is given to RL agent
        - 11.12.23: INVENTORY_MODE as global variable
        - 11.12.23: eoq_dynamic is calculated in the Demand class
        
    Comments from last version:
        - form V1.1 it uses the same quantity regime as planning uses
        - 06.07.23: added n_put and n_get and related functionality
        - 21.11.23: added some comments to describe what we actually do + minor bug fix on ROP
    
    Possible problems:
        - __init__ function for calculating eoq_dynamic the whole actual demand is taken. 
            Maybe it is not allowed.
    """

    def __init__(self, class_Demand):
        self.demand_class = class_Demand          # demand class
        self.product = class_Demand.product

        # Intialize the inventory level depending on the mode
        if INVENTORY_MODE == 0:
            # No initial amount
            self.inventory_level = [0]
        elif INVENTORY_MODE == 1:
            # Initial amount = safety stock
            self.inventory_level = [int(self.product["safety_stock"])]
        elif INVENTORY_MODE == 2:
            # Initial amount = safery stock + static eoq
            self.inventory_level = [int(self.product["safety_stock"] + self.demand_class.eoq)]

        # ADD HERE MORE n_put and n_get if necessary


    @property
    def avg_level(self):
        """
        Calculates the average inventory level
        """
        
        return np.mean(self.inventory_level)


    def put(self, qty):
        """
        Adds a given amount to the inventory
        """
        
        assert qty >= 0, "Quantity must be positive"
        self.inventory_level.append(self.inventory_level[-1] + qty)


    def get(self, qty):
        """
        Subtracts a given amount from the inventory
        """

        assert self.inventory_level[-1] >= qty, "Quantity amount must be smaller than inventory level"
        assert qty >= 0, "Quantity must be positive"

        # ADD HERE CODE checking if amount equals to 0 to return nothing IF NECESSARY

        #self.inventory_level.append(self.inventory_level[-1] - qty)
        self.inventory_level[-1] = self.inventory_level[-1] - qty


class Warehouse:
    """
    This class implements the warehouse for both products.
    It is used for checking the maximum capacity of the warehouse, which is the sum of 
        both inventory levels of the products.
    
    Changes:
        - 27.12.23: added max inventory variable and related functionality
        - 29.12.23: added a function for checking if it is possible to store an amount in inventory
    """

    def __init__(self, p1_class_Inventory, p2_class_Inventory) -> None:
        self.products = (p1_class_Inventory, p2_class_Inventory)

    
    @property
    def current_warehouse_level(self):
        """
        This function calculates the current inventory in the warehouse.
        """
            
        return self.products[0].inventory_level[-1] + self.products[1].inventory_level[-1]


    def is_possible_to_store(self, order):
        """
        This function checks if it is possible to store a given amount for a product in the warehouse.
        """

        if self.current_warehouse_level + order["amount"] <= MAXIMUM_INVENTORY:
            return True
        else:
            return False



class Machine:
    """
    This class is for a production plan and execution for one machine.

    Changes:
        - 28.12.23: changed the class completely. For more info, see readme
        - 11.12.23: No reorder point is considered. No eoq. 
        - 11.12.23: TODO: MAYBE NO SCHEDULING 


    Steps TO BE DELETED LATER:
        1. Calculate the total time needed for a production order
        2. If the calculated time is smaller than the time left in a day, 
            the order can't be executed in a day. Therefore, a smaller amount is produced.
        3. RULE: it is not possible to share a production order between days.
            The agent has to learn itself how much to give to the machine.
        Every day, the machine is given a production order. It assesses if it can be produced
            in that day by comparing the time left in a day and the time needed for the order.
    """

    def __init__(self, class_Warehouse) -> None:
        self.warehouse = class_Warehouse
        self.products = class_Warehouse.products
        self.total_time_day = SEC_PER_DAY #previous: MACHINE_WORKING_HOURS * SEC_PER_HOUR
        self.current_product = None
        self.last_product = None
        self.current_order = None #keeps track of the current order
        # ADD HERE MORE

    
    @property
    def product_number(self):
        """
        This function returns which product (id: 0 or 1) is currently produced/processed.
        """

        if self.current_order == None:
            raise ValueError("Give first an order to the machine")      
        elif self.current_order["product"] == "p1":
            return 0
        elif self.current_order["product"] == "p2":
            return 1
    

    def is_possible_to_produce(self, order):
        """
        This function checks if it is possible to produce a given order in a day
            by comparing the time left in a day and the time needed for the order.
        Check the setup time too, if the product produced on the previous day is different.
        
        The structure of variable order:
            order = {
                "product": "p1" or "p2",
                "amount": amount as integer
            }
        
        """ 

        #store the current order
        self.current_order = order
        # set the current product
        self.current_product = self.products[self.product_number]
        
        order_duration = lognorm_int(
                            mu = order["amount"] * self.current_product.product["t_e"],
                            varcoef = MACHINE_VARCOEF / np.sqrt(order["amount"]),
                            round = False
                        ) * SEC_PER_DAY
                        #using just the function gives the total time in days to produce an order. 
                        #It is necessary to multiply by number of seconds in a day.
        
        # calculate the setup time
        setup_time = lognorm_int(
                        mu = self.current_product.product["t_r"],
                        varcoef = MACHINE_VARCOEF,
                        round = False
                    ) * SEC_PER_DAY if self.last_product != order["product"] else 0
                    # the same applies here as mentioned in the previous lognorm calculation
        # store the last product
        self.last_product = order["product"]

        # compare if total duration is smaller than the time left in a day
        if order_duration + setup_time <= self.total_time_day:
            return True
        else:
            return False
    

    def produce(self, order):
        """
        This function checks first if it is possible to produce a given order in a day.
        Then it checks if it is possible to store the produced quantity.
        If both conditions are met, the order is produced and stored.
        """

        #store the current order
        self.current_order = order

        # check if it is possible to produce the order
        if self.is_possible_to_produce(order):
            # check if it is possible to store the produced quantity
            if self.warehouse.is_possible_to_store(order):
                # produce the order and store it (remark: no "production" done here. It is assumed done in self.is_possible_to_produce)
                self.warehouse.products[self.product_number].put(order["amount"])
                # to have the same length of lists for both inventories, add 0 to the other inventory
                self.warehouse.products[1 - self.product_number].put(0)
                # return 0 if the order was produced
                return 0 #The order has been produced and stored successfully
            else:
                # return 101 if the order was not produced
                return 101 #The order has been produced but not stored
        else:
            # return 201 if the order was not produced
            return 201 #The order has not been produced
    

    def fulfill(self, t):
        """
        This function fulfills the demand for a current product.

        TODO: 
            - Change later the variable t to a global one for training the RL agent.
            - Move it to Warehouse class (not mandatory)
        """
        
        # provide all material to demand and deduct the consumed amount from the inventory
        max_material = self.warehouse.products[self.product_number].inventory_level[-1]
        used_material, exit_code = self.warehouse.products[self.product_number].demand_class.fulfill(max_material, t)
        # deduct the used material from the inventory
        self.warehouse.products[self.product_number].get(used_material)
        # to have the same length of lists for both inventories, add 0 to the other inventory
        self.warehouse.products[1 - self.product_number].get(0)
        # return the exit code
        return exit_code