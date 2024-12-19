import random

def objective_function(x):
    return sum(x_i ** 2 for x_i in x)

class Particle:
    def __init__(self, dimension, bounds):
        self.position = [random.uniform(bounds[0], bounds[1]) for _ in range(dimension)]
        self.velocity = [random.uniform(-1, 1) for _ in range(dimension)]
        self.pBest = list(self.position)
        self.pBest_fitness = objective_function(self.position)

class PSO:
    def __init__(self, dimension, bounds, num_particles=30, max_iterations=100):
        self.dimension = dimension
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.particles = [Particle(dimension, bounds) for _ in range(num_particles)]
        self.gBest = list(self.particles[0].position)
        self.gBest_fitness = self.particles[0].pBest_fitness
        self.w = 0.5  
        self.c1 = 1.5   
        self.c2 = 1.5  

    def optimize(self):
        for iteration in range(self.max_iterations):
            for particle in self.particles:
                fitness = objective_function(particle.position)

                # Update personal best (pBest)
                if fitness < particle.pBest_fitness:
                    particle.pBest = list(particle.position)
                    particle.pBest_fitness = fitness

                # Update global best (gBest)
                if fitness < self.gBest_fitness:
                    self.gBest = list(particle.position)
                    self.gBest_fitness = fitness

            # Update velocity and position for each particle
            for particle in self.particles:
                for i in range(self.dimension):
                    # Update velocity
                    r1, r2 = random.random(), random.random()
                    particle.velocity[i] = (self.w * particle.velocity[i]
                                            + self.c1 * r1 * (particle.pBest[i] - particle.position[i])
                                            + self.c2 * r2 * (self.gBest[i] - particle.position[i]))
                    # Update position
                    particle.position[i] += particle.velocity[i]

                    # Ensure position stays within bounds
                    particle.position[i] = max(self.bounds[0], min(particle.position[i], self.bounds[1]))

        return self.gBest, self.gBest_fitness

# Define parameters
dimension = 2  # Number of dimensions
bounds = (-10, 10)  # Search space bounds for each dimension
num_particles = 30  # Number of particles in the swarm
max_iterations = 100  # Maximum number of iterations
print('Raghavendra R, 1BM22CS214')

# Create PSO instance and optimize
pso = PSO(dimension, bounds, num_particles, max_iterations)
best_position, best_fitness = pso.optimize()

# Output the result
print(f"Best Position: {best_position}")
print(f"Best Fitness: {best_fitness}")


'''
1. objective function 
2. intialize the particle position and velocity 
3. update the velocity and the position
4. evaluate function, if score < best_Score, update it and return best score and best position.
'''