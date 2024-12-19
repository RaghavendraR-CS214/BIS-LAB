import numpy as np

def objective_function(x):
    return x ** 2  # The function to minimize

def initialize_wolves(num_wolves, search_space):
    return np.random.uniform(search_space[0], search_space[1], num_wolves)

def update_position(alpha, beta, delta, wolf, a):
    r1, r2 = np.random.rand(), np.random.rand()
    A = 2 * a * r1 - a
    C = 2 * r2
    D = abs(C * alpha - wolf)
    X1 = alpha - A * D

    r1, r2 = np.random.rand(), np.random.rand()
    A = 2 * a * r1 - a
    C = 2 * r2
    D = abs(C * beta - wolf)
    X2 = beta - A * D

    r1, r2 = np.random.rand(), np.random.rand()
    A = 2 * a * r1 - a
    C = 2 * r2
    D = abs(C * delta - wolf)
    X3 = delta - A * D

    return (X1 + X2 + X3) / 3

def grey_wolf_optimization(obj_func, num_wolves=5, max_iter=50, search_space=(-10, 10)):
    # Initialize wolves' positions
    wolves = initialize_wolves(num_wolves, search_space)
    fitness = np.array([obj_func(wolf) for wolf in wolves])

    # Identify alpha, beta, delta
    sorted_indices = np.argsort(fitness)
    alpha, beta, delta = wolves[sorted_indices[0]], wolves[sorted_indices[1]], wolves[sorted_indices[2]]

    a = 2  # Initial value for the parameter a

    for iteration in range(max_iter):
        for i in range(num_wolves):
            wolves[i] = update_position(alpha, beta, delta, wolves[i], a)
            wolves[i] = np.clip(wolves[i], search_space[0], search_space[1])  # Ensure wolves stay within bounds

        # Recalculate fitness and update alpha, beta, delta
        fitness = np.array([obj_func(wolf) for wolf in wolves])
        sorted_indices = np.argsort(fitness)
        alpha, beta, delta = wolves[sorted_indices[0]], wolves[sorted_indices[1]], wolves[sorted_indices[2]]

        # Decrease a linearly
        a = 2 - (2 * (iteration / max_iter))

        # print(f"Iteration {iteration+1}: Alpha = {alpha}, Fitness = {obj_func(alpha)}")

    return alpha, obj_func(alpha)

# Run the algorithm
print("Raghavendra R, 1BM22CS214")
best_position, best_fitness = grey_wolf_optimization(objective_function)
print(f"Best Position: {best_position}")
print(f"Best Fitness: {best_fitness}")