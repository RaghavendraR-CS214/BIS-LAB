import numpy as np
import math

# Objective function (example: Sphere function, you can replace it)
def objective_function(x):
    return sum(x**2)  # Minimize the sum of squares


def levy_flight(beta, d):
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)))**(1 / beta)
    u = np.random.normal(0, sigma_u, d)  # Draw from Gaussian distribution
    v = np.random.normal(0, 1, d)
    step = u / (abs(v)**(1 / beta))
    return step

# Cuckoo Search Algorithm
def cuckoo_search(n, d, alpha, pa, maxGen):
    # n: Population size, d: Dimension of the problem
    # alpha: Step size, pa: Discovery probability, maxGen: Max iterations
    
 
    nests = np.random.uniform(-10, 10, (n, d)) 
    fitness = np.array([objective_function(nest) for nest in nests])  
    
   
    best_nest_index = np.argmin(fitness)
    best_nest = nests[best_nest_index]
    best_fitness = fitness[best_nest_index]
    
    beta = 1.5  
    
    # Step 2: Iterative loop
    for gen in range(maxGen):
        for i in range(n):
            # Generate a new solution via Lévy flight
            step = levy_flight(beta, d)
            new_nest = nests[i] + alpha * step * (nests[i] - best_nest)
            new_nest = np.clip(new_nest, -10, 10)  # Keep solutions within bounds
            
            # Evaluate new fitness
            new_fitness = objective_function(new_nest)
            if new_fitness < fitness[i]:  # Replace with better solution
                nests[i] = new_nest
                fitness[i] = new_fitness
        
        # Abandon some nests with a probability pa
        for i in range(n):
            if np.random.rand() < pa:
               # Replace with new random solution
                nests[i] = np.random.uniform(-10, 10, d)
                fitness[i] = objective_function(nests[i])
        
        # Update the current best
        best_nest_index = np.argmin(fitness)
        if fitness[best_nest_index] < best_fitness:
            best_nest = nests[best_nest_index]
            best_fitness = fitness[best_nest_index]
        
        # print(f"Generation {gen+1}, Best Fitness: {best_fitness:.5f}")
    
    
    return best_nest, best_fitness


n = 25         
d = 5         
alpha = 0.01   
pa = 0.25      
maxGen = 100   


print('Raghavendra R, 1BM22CS214')
best_solution, best_value = cuckoo_search(n, d, alpha, pa, maxGen)
print("Best Solution:", best_solution)
print("Best Fitness Value:", best_value)