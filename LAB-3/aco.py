import numpy as np
import random

class ACO:
    def __init__(self, n_ants, n_iterations, alpha, beta, rho, deposit, cities):
        self.n_ants = n_ants  # Number of ants
        self.n_iterations = n_iterations  # Number of iterations
        self.alpha = alpha  # Pheromone influence
        self.beta = beta  # Distance influence
        self.rho = rho  # Evaporation rate
        self.deposit = deposit  # Pheromone deposit constant
        self.cities = cities  # Coordinates of cities
        self.n_cities = len(cities)  # Number of cities
        self.distances = self.calculate_distances()  # Distance matrix
        self.pheromones = np.ones((self.n_cities, self.n_cities))  # Initial pheromones

    def calculate_distances(self):
        """Calculate Euclidean distances between cities."""
        distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                distances[i][j] = np.linalg.norm(np.array(self.cities[i]) - np.array(self.cities[j]))
        return distances

    def construct_solution(self):
        """Construct a solution (path) for one ant."""
        path = [random.randint(0, self.n_cities - 1)]  # Random starting city
        while len(path) < self.n_cities:
            current_city = path[-1]
            next_city = self.choose_next_city(current_city, path)
            path.append(next_city)
        return path

    def choose_next_city(self, current_city, path):
        """Choose the next city based on pheromone and distance."""
        probabilities = []
        for next_city in range(self.n_cities):
            if next_city not in path:
                pheromone = self.pheromones[current_city][next_city] ** self.alpha
                visibility = (1 / self.distances[current_city][next_city]) ** self.beta
                probabilities.append(pheromone * visibility)
            else:
                probabilities.append(0)
        probabilities = np.array(probabilities) / sum(probabilities)
        return np.random.choice(range(self.n_cities), p=probabilities)

    def calculate_distance(self, path):
        """Calculate the total distance of a path."""
        return sum(self.distances[path[i - 1]][path[i]] for i in range(len(path)))

    def update_pheromones(self, paths, distances):
        """Update pheromones based on the paths taken by ants."""
        self.pheromones *= (1 - self.rho)  # Evaporation
        for path, distance in zip(paths, distances):
            pheromone_contribution = self.deposit / distance
            for i in range(len(path)):
                self.pheromones[path[i - 1]][path[i]] += pheromone_contribution

    def run(self):
        """Run the ACO algorithm."""
        best_path = None
        best_distance = float('inf')
        for _ in range(self.n_iterations):
            paths = [self.construct_solution() for _ in range(self.n_ants)]
            distances = [self.calculate_distance(path) for path in paths]
            self.update_pheromones(paths, distances)
            # Update the global best path
            for path, distance in zip(paths, distances):
                if distance < best_distance:
                    best_path, best_distance = path, distance
        return best_path, best_distance


# Example Usage
if __name__ == "__main__":
    # Random city coordinates
    cities = [(0, 0), (2, 3), (5, 5), (8, 1)]
    aco = ACO(n_ants=10, n_iterations=100, alpha=1, beta=2, rho=0.1, deposit=100, cities=cities)
    best_path, best_distance = aco.run()
    print("Raghavendra R, 1BM22CS214")
    print("Best Path:", best_path)
    print("Best Distance:", best_distance)
