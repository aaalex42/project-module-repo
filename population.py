
'''
the create_population Function use to creat the population that will be used in Genetic algorithm.
the poulation selected based on optimization between setup_cost and holding cost and all value ensure to cover the required demand at that time
'''


def create_population(population_pipeline):
    all_product_population_list=[]
    
    for product in population_pipeline: #loop to create the population for all products
            
            #get the value related to each product
            previous_cost=0
            setup_cost=product['setup_cost']
            holding_cost_per_month=product['holding_cost_per_month']
            setup_time=product['setup_time']
            production_time=product['Production_time']
            total_list=[]
            forecast=product['forecast']
            current_demand=forecast[0]
            Current_inv=product['Current_inv']
            product_object=product['product_object']
            
            
            #if the demand start with zeros we skip them and update the demand
            for i,value in enumerate(forecast):
                if value==0:
                       continue
                else:
                    forecast=forecast[i:]
                    break 
            
            demand=forecast
            product_name=product['product']
            Population_list=[]
            previous_amount=0
            
            for i in range(len(forecast)):# for loop to create population depending on setup_cost and holding cost
                
                    if forecast[i]==0:#skip zero value demand
                        continue
                    
                    holding_cost=i*holding_cost_per_month*forecast[i]#calculate the holding cost 
                    if i==0:#add the setup cost 
                        total_cost=previous_cost+setup_cost
                    total_cost+=holding_cost#update the total cost each loop
                    
                    total_list.append(total_cost)#store the costs in one list
                    #store the population (the suggested producing amount  with index i related to expected order day
                    Population_list.append((i,forecast[i]+previous_amount))
                    #update the previous_amount
                    previous_amount+=forecast[i]
                    if holding_cost>=setup_cost :
                                total_list=total_list[:-1]# the total cost list
                                previous_cost=total_list[-1] #update the previous cost
                                forecast=forecast[i:]#update the next demand list for next calculation 
                                #calculate the cost ,production time for each suggested production amount
                                product_amount_cost_time=calculate_max_optimal_cost_time(demand,Population_list[:-1],setup_cost,
                                                           holding_cost_per_month,
                                                           setup_time,production_time,product_name,current_demand,Current_inv,product_object)
                                all_product_population_list.append(product_amount_cost_time)
                                                                          
                                break
                
                
                
    return all_product_population_list  
'''This function used to return the maximum optimal cost if we produce the given amount of each product. the cost calculted based on produce as much as possible without exceed the setup cost for each batch. we start with first batch till the holding cost equals or greater than the setup cost then we stop and start new batch until we produce all demand '''   
def calculate_max_optimal_cost_time(demand,population,setup_cost,holding_cost,setup_time,production_time,product_name,current_demand,Current_inv,product_object):
    
    
    
    
    All_demand=demand
    current_demand=current_demand
    Current_inv=Current_inv
    production_time=production_time
    setup_time=setup_time
    amount_cost_list=[]
    setup_cost=setup_cost
    holding_cost_per_month=holding_cost
    population_list=population
    product_name=product_name
    product_object=product_object
    
    
    
    for i in range(len(population_list)):
        previous_cost=0
        current_amount=population_list[i][1]
        
        index_in_demand=population_list[i][0]
        
        
        #calculate the previous cost
        previous_range=All_demand[:index_in_demand+1]#always start with non_zero value
        for i in range(len(previous_range)):
            if i ==0:
                previous_cost=setup_cost
            previous_cost+=i*holding_cost_per_month*previous_range[i]
        #the next demand to produce 
        forecast=All_demand[index_in_demand+1:]
        #if the demand start with zeros we skip them anf update the demand
        for i,value in enumerate(forecast):
                if value==0:
                       continue
                else:
                    forecast=forecast[i:]
                    break 
        #print("next amount to produce",forecast)#always non zero_value
        total_list=[]
        len_search=len(forecast)
        ##########################################
        for x in range(len_search):
            for i in range(len(forecast)):
             
                holding_cost=i*holding_cost_per_month*forecast[i]
                if i==0:
                    total_cost=previous_cost+setup_cost
                total_cost+=holding_cost
                total_list.append(total_cost)
                
                if holding_cost>=setup_cost :#when the holding cost greater than setup cost we stop producing.
                    total_list=total_list[:-1]#update the total_cost list,remove the last value
                    previous_cost=total_list[-1]#update the previous cost that will be used in next loop
                    forecast=forecast[i:]#update the next demand to produce
                    
                    break
            if len(total_list)==len_search : 
                
                #print("the optimal total cost for the whole demand is:",total_list[-1],"dollar")
                total_production_time=current_amount*production_time+setup_time
                amount_cost_list.append((product_name,current_amount,total_list[-1],total_production_time, setup_time,production_time,current_demand,Current_inv,product_object))
                break
    #print("amount_cost_list",amount_cost_list)
    return amount_cost_list
    
        
    
        
    
 