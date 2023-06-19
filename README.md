<img src="https://www.thws.de/fileadmin/_processed_/3/5/csm_header_A-Grubnyak-Unsplash-02_bc7c4600d2.png"  title="CAIRO Logo">

# Project Module Repository

**Improvement of Production Planning Techniques using ML**

---

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