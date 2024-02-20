'''The Genetic Algorithm used to provide us with final solution, How much should we produce from each product to Satisfy 
        the following conditions:
        1. not exceed the max capacity of warehouse.
        2.both product should be fulfilled 
        3. the combination of both product should be at the minimum cost between 
            the suggested Solution that we got in population_set
        
        '''



import random
import datetime
def get_fitness(guess,Avai,limit): # get us how much the guess(Suggested production job) is fit
    Avai=Avai # the machine availablility
    amount=0# the initial amount
    
    #A.check the size limit (the value A either 0: exceed the limit, 1:in the limit)
    for i in guess:
        amount+=i[1]+i[7]#the sum of suggested produce amount + current inventory level
  
    if amount<= limit:
        A=1
    else: A=0
        
    #B.check fullfillment
    
    F={'F1': 0, 'F2': 0}# initial Value of the Fulfilment 
    for i in guess:
        
        if Avai>0:#check if the machine available 
            if i[3]<=Avai:# the amount of Production is less than or equal the available machine time
                F[f'F{i[0]}']=1 # we produce the whole amount
                Avai=Avai-i[3]#update the available time
            else:#partial amount to produce today
                produced_amount_today=(Avai-i[4])/i[5]#calculate how many items can be produced today
                Avai=0# the machine not available anymore 
                if (produced_amount_today+i[7])>= i[6]:#if produced amount+ current inv greater than current demand
                    F[f'F{i[0]}']=1

        else:
            break
    
    Fu=F['F1']*F['F2']# calculate the final fulfilment factor for all product, both should be fulfilled 
    #C.Calculate the total cost 
    C=guess[0][2]+guess[0][2]# the sum of expexted cost for both products 
    return_value=A*Fu*C
    #if the guess exceed the limit of capacity or the both products are not fullfillied, the function return big value.
    if Fu==0 or A==0:
        return_value=10000000000 #big value
    
     
    return return_value # return the final fitness value

def display(guess,startTime,Avai,limit):# display the time for each Generation,and the suggested amount from each products.
    timeDiff = datetime.datetime.now() - startTime
    fitness = get_fitness(guess,Avai,limit)
    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(guess[0][0],guess[0][1],guess[1][0],guess[1][1], fitness, str(timeDiff)))
def generate_parent(geneSet):# only one parent'first one in the family'
        
        parent=random.sample(geneSet,1)# create it as list
        parent_1=parent[0]# we need it as tuple
        
        return parent_1
def creat_geneSet(x):
    
   
    from itertools import product
    if len(x)==2:
                
                # Assuming 'list_one' and 'list_two' are already defined as your first and second lists respectively
                list_one = x[0]  # Add your actual data
                list_two = x[1]

                # Generate permutations with items from both lists in both directions
                cross_list_permutations = list(product(list_one, list_two))
                cross_list_permutations_reversed = list(product(list_two, list_one))

                 # Combine both sets of permutations
                combined_permutations = cross_list_permutations + cross_list_permutations_reversed

            # Now, 'combined_permutations' is a list containing all the permutations
            # You can access and use this list directly in your code
    else:
        combined_permutations=x
        
    return combined_permutations
def mutate(parent,geneSet): 
    # create new Solution under the condition that should be diffrent than the old one
    previous_solution=parent
    new_guess, alternate = random.sample(geneSet, 2) #generate 2 random sample from geneset
    
    #update the random parent gene with diffrent random gene
    final_guess = alternate if new_guess == previous_solution else new_guess
    return final_guess
def get_final_result(population_set,Avai,limit,inv_class_dic,job_list,now,demand_object_list):
    print(Avai)
    geneSet=creat_geneSet(population_set)
    startTime = datetime.datetime.now()
    current_best_Parent = generate_parent(geneSet)#initial value
    #print("current_best_Parent",current_best_Parent)
    current_max_Fitness = get_fitness(current_best_Parent,Avai,limit)#initial value
    #print(current_max_Fitness,"current_max_Fitness")
    display(current_best_Parent,startTime,Avai,limit)
    for i in range(500):
        
        child = mutate(current_best_Parent,geneSet)
        childFitness = get_fitness(child,Avai,limit)#get the rank of new generation
        #print(childFitness)
        if current_max_Fitness <= childFitness :
 
            continue#skip the rest of the code inside the loop and move on to the next iteration.
        display(child,startTime,Avai,limit)
        
        current_max_Fitness = childFitness
        current_best_Parent = child
        # create the production job
    if current_max_Fitness!=10000000000:# check if we got guess satisfy our criteria 
        job_list=[]
        #set the limit for each product that should not go lower than it
        product_one_minimum_limit=300000
        product_two_minimum_limit=7000000
        for i in current_best_Parent:
            
            if i[0]==1 and i[7]<=product_one_minimum_limit:
                
                job_list.append({'amount':  i[1],'date':now , 'inv': inv_class_dic[f'inv{i[0]}'], 'p': i[8]})
                demand_object_list[0].n_jobs += 1#product one
            if i[0]==2 and i[7]<=product_two_minimum_limit:
                job_list.append({'amount':  i[1],'date':now , 'inv': inv_class_dic[f'inv{i[0]}'], 'p': i[8]})
                demand_object_list[1].n_jobs += 1#product two
          
    return job_list 



          
