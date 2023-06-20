### Class production plane
Production plane and execution for one machine.
Implements : planning step including reorder point,econimic quantity and order scheduling /actual scheduling/production step.

+plan:
adds production jobs,"produce" routine eliminates them again.It Hazty with adding new orders (generating new demand) then takes demand /inventory pipeline and fills the job pipline.

1.iterate over the demands/inventories queue.

1.1. Seperate demand and inventory and product in different varriattes.
1.2. Generate demand and forcast for 1 day:
If the current inventory level for a poduct is <= than the reorder point and replenishment due of a product is <= than 0.
If replenishment open of a product is greater than  o THEN  calculate the EOQ dynamic with the actual demand (only first element ) and the next forcasts.
for the calculation cost per piece (for eoq), cost per batch and inventory interest (and eoq mode)

If eoq > 0 
      THEN add a new element to the joblist containg anount =eoq, current date ,inventory (current ) and relevant product.
      2.decrease the replenishment open by the eoq.
2.Schedule the created joblist

+schedule:
Schedule the joblist using diffrent methods:Value,SPT,LPT, Slip

If scheduling regime= Value
 the joblist is produced not in the ascending order ,but the joblist is then sorted by value = multiplying amont by cost per piece for eoq.
If scheduling regime = SPT
  ordering according to Time.
  the joblist is sorted by value= multiplying  the amount by production time per piece 
If scheduling regime = LPT
  Ordering similer to SPT but in the opposite direction.
If scheduling regime = Slip
   the joblist is sorted by value =substracting  the date by (amount multiplied by production time per piece)
   sort the joblist by sortes and ascending value
   Eliminate sortes.
+Produce

executes the first joblist from the list and eliminated the entry. it return the duration of the production job , amount of parts produced and inventory.
  If length of the Joblist is 0, return 0,0,0 for duration , amount and inventory.
place the amount, inventory and product in seprate variates.
production amount must be greater than 0 .
Calculate the duration to produce the given amount by sampling from Lognorm distribuation with mean =amount*production time per piece and sigma=varcoef/ √(amount)

*Increas the production time (total) by the sampled duration.
*calculate the setup time by sampling from Lognorm Disribuation with the mean setup time per batch and sigma varcoef (of a machine).
*But only in the Case if last product is diffrent than the current one (no setup time nedded if the product is the same).
*Increase the duration of production by the setup time .
*add the setup time to the total setup time .
*update the last product variable.
Delete the first element from the joblist.
Return the calculated duration of production Job, the amount produced and inventory.


