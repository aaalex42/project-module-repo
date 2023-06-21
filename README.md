<img src="https://www.thws.de/fileadmin/_processed_/3/5/csm_header_A-Grubnyak-Unsplash-02_bc7c4600d2.png"  title="CAIRO Logo">

# Project Module Repository

**Improvement of Production Planning Techniques using ML**

---


## Class Demand
Contains several important Functions: 
    - Generation of Demand and Forcast
    - Fulfillment of Demand
    - Service Level
    - average daily machine time

initialze product, rbase and d_fc
len(demand list) - 2*len(forecast list)

### **Function *generate demand and forecast***
    gen_demand_and_fc:
        distribution_int: creates a random variable following lognorm distribution with *d_mue* and *d_varcoef* and appends to demand
        **only** if the pointer to current time is equal to the next event
        **else** demand is 0
        **also** the next event is generating poisson distribution creating a random vaariable with d_t_mu(number of day) an average between orders
        **until here** is demand generated

nextd - next event
in days where demand is generated

    part0: generate a random number using lognorm with a d_mue and d_varcoef
    part1: from that random number subtract the mean Qty/2
    part2: add the generated demand with part 1*forcast_noise
    part3: round to main r_base
    final part: add the generated forecast to the list by taking first the max between 0 and the generated forecast


### **Function *fulfill***

**Input:** *material, allow_part_shipment*

**Output:** *list of demanded amounts*


    fulfill(mat, allow_part_shipments):
        return material consumed for this step at the beginning, mat_used = 0
        b10 = backlog
    1. handle the backlog as much as possible if material handed down to manage the demand stream ->= backlog
    Then increase the mat_used by backlog decrease the material amount by backlog set backlog = 0
    Else (if backlog is bigger or there is no material increaste the material used by material
    decrease the backlog by material  and set material = 0 (all material has been consumed))
    2. Then handle the current damnd step and update statistics
        **if** the remaining material >= current demand 
        **THEN** increase the successful demand fulfillments by 1 increase the material used by the current demand
        **ELSE** if there is no fulfillment remaining material, increase failed demand fulfillments by 1
        **if** part shipments are allowed
            then increase the backlog by the difference between current demand and available material
            Increase the used material by the remaining material + set the current material to 0
        **ELSE**(if part shipments are not allowed increase the backlog by the current demand)

    3.  Get rid of the current demand from buffer and save stats
        - Add the current demand to the list of demanded amounts
        - Append the total material used to the list of fulfillments
        - Append the difference between the initial backlog and final backlog to the list of delta backlogs
        - delete the first element from the list of demands which is the current demand


### **Function *average daily machine time***

returns the expected daily machine time consumed by the product:

    1. calculate the daily demand of a product by the ratio/division of the Quantity per Day/number of days on average between 2 orders
    2. Calculate the daily production time by multiplying the daily demand by production time per piece in fractions of day
    3. Calculate EOQ dynamic
    4. calculate the daily setup time by dividing daily demand by eoq and multiplying setup time per batch in fractions of a day
    5. Final step: return the sum of daily production time + daily setup time

### Outside of Demand Class

#### **Function *eoq_static***
returns the andler forumla (constant demand)
Calculates the Andler formula

#### **Function *eoq_dynamic***

calculates either by andler formular on fixed order period

    if mode is chosen andler
        1. calculate the sum of all demands from demand list
        2. calculate the lenght of the demand list
        3. calculate the annual demand by multiplying average demand per day and number of days per year
        4. use the above mentioned formula for static eoq
    if mode is FOP:
        1. calculate the minimum between FOP periods and the length of the demand list and then subtract
        2. calculate the sum of the demand list but from the first element until the index calculated in step1

## Class Inventory


### **Function *avg_level***

### **Function *level***

### **Function *put***

### **Function *get***

---

## Process Wrappers



### **Function *produce***

**Input:** *pp ProdPlan*

**Output:** *None*

It is a process wrapper around production plan, execution part and inventory update.
It runs a production plan (i.e., one machine). Also, it has no bucket, as it continuously processes all jobs pending.

```
WHILE run forever (or until given simulation cycles):
    IF duration of a job > 0:
        THEN:
            Wait this duration to pass
            Store the product after the production is finished (no earlier).
            Decrease the replenishment due by amount. 
        ELSE:
            Wait 15 minutes to prevent infinite loop.
```

### **Function *fulfill***

**Input:** *d Demand, inv Inventory, allow_part_shipments Bool, bucket (**CHECK VAR TYPE**)*

**Output:** *None*

It is a demand fulfillment function for one set of demand and inventory per bucket.

```
WHILE run forever (or until given simulation cycles):
    Get the current inventory level.
    Fulfill the current inventory level (from Demand class).
    Subtract the current inventory level from the inventory.
    Yield timeout Bucket = 1 day (CHECK HERE FOR MORE INFO)
```

> Yield timeout Bucket = 1 day (CHECK HERE FOR MORE INFO)

### **Function *plan***

**Input:** *pp ProdPlan*

**Output:** *None*

Run one planning cycle per bucket (i.e., for 1 machine)

```
WHILE run forever (or until given simulation cycles):
    Plan (from ProdPlan class) at the current time step
    Yield timeout Bucket = 1 day (CHECK HERE FOR MORE INFO)
```

> Yield timeout Bucket = 1 day (CHECK HERE FOR MORE INFO)

---

The process wrappers are executed in the following order:

1. Launch the fulfillment for each product
2. Run planning for a machine
3. Run production for a machine