"""
Genetic Algorithm for the delivery-route problem (fictional dataset).
Depot = A. Deliveries = B, C, D, E, F, G. Route must start and end at A.
"""

import random

# ---------------------------------------------------------------
# 1. Problem data: symmetric distance matrix (kilometres)
# ---------------------------------------------------------------
LOCATIONS = ["A", "B", "C", "D", "E", "F", "G"]
DISTANCES = {
    "A": {"A": 0, "B": 4, "C": 8, "D": 6, "E": 7, "F": 9, "G": 5},
    "B": {"A": 4, "B": 0, "C": 5, "D": 7, "E": 3, "F": 8, "G": 6},
    "C": {"A": 8, "B": 5, "C": 0, "D": 4, "E": 6, "F": 3, "G": 7},
    "D": {"A": 6, "B": 7, "C": 4, "D": 0, "E": 5, "F": 6, "G": 8},
    "E": {"A": 7, "B": 3, "C": 6, "D": 5, "E": 0, "F": 4, "G": 6},
    "F": {"A": 9, "B": 8, "C": 3, "D": 6, "E": 4, "F": 0, "G": 5},
    "G": {"A": 5, "B": 6, "C": 7, "D": 8, "E": 6, "F": 5, "G": 0},
}
DEPOT = "A"
STOPS = ["B", "C", "D", "E", "F", "G"]  # a chromosome is a permutation of these

# ---------------------------------------------------------------
# 2. GA parameters
# ---------------------------------------------------------------
RANDOM_SEED = 42        # fixed seed -> reproducible experiment
POPULATION_SIZE = 20
GENERATIONS = 60
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.30
ELITE_COUNT = 2         # best routes copied unchanged to next generation


# ---------------------------------------------------------------
# 3. Fitness: total distance of A -> chromosome -> A (smaller = better)
# ---------------------------------------------------------------
def route_distance(chromosome):
    full_route = [DEPOT] + chromosome + [DEPOT]
    total = 0
    for i in range(len(full_route) - 1):
        total += DISTANCES[full_route[i]][full_route[i + 1]]
    return total


def format_route(chromosome):
    return " -> ".join([DEPOT] + chromosome + [DEPOT])


# ---------------------------------------------------------------
# 4. Initial population: random valid permutations of B-G
# ---------------------------------------------------------------
def create_initial_population(size):
    population = []
    for _ in range(size):
        chromosome = STOPS.copy()
        random.shuffle(chromosome)
        population.append(chromosome)
    return population


# ---------------------------------------------------------------
# 5. Parent selection: tournament selection
#    Pick a few random routes and keep the shortest one.
# ---------------------------------------------------------------
def tournament_selection(population):
    contestants = random.sample(population, TOURNAMENT_SIZE)
    return min(contestants, key=route_distance).copy()


# ---------------------------------------------------------------
# 6. Route-preserving crossover (Order Crossover, OX)
#    Copy a slice from parent 1, fill the remaining positions with the
#    missing locations in the order they appear in parent 2.
#    Result: every location appears exactly once.
# ---------------------------------------------------------------
def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]

    remaining = [stop for stop in parent2 if stop not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = remaining[idx]
            idx += 1
    return child


# ---------------------------------------------------------------
# 7. Mutation: swap two randomly chosen locations
# ---------------------------------------------------------------
def swap_mutation(chromosome):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome


# ---------------------------------------------------------------
# 8. Validity check (start/end at depot handled by route_distance)
# ---------------------------------------------------------------
def is_valid(chromosome):
    return sorted(chromosome) == sorted(STOPS)


# ---------------------------------------------------------------
# 9. Main GA loop
# ---------------------------------------------------------------
def run_ga():
    random.seed(RANDOM_SEED)
    population = create_initial_population(POPULATION_SIZE)

    best_route = min(population, key=route_distance)
    best_distance = route_distance(best_route)

    for generation in range(1, GENERATIONS + 1):
        # Sort so the shortest routes come first
        population.sort(key=route_distance)

        # Elitism: copy the best routes unchanged
        next_population = [r.copy() for r in population[:ELITE_COUNT]]

        # Fill the rest with children
        while len(next_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = order_crossover(parent1, parent2)
            child = swap_mutation(child)
            next_population.append(child)

        population = next_population

        current_best = min(population, key=route_distance)
        current_distance = route_distance(current_best)
        if current_distance < best_distance:
            best_route, best_distance = current_best.copy(), current_distance

        if generation == 1 or generation % 10 == 0:
            print(f"Generation {generation:3d} | best distance so far: "
                  f"{best_distance} km")

    return best_route, best_distance


if __name__ == "__main__":
    print(f"Seed={RANDOM_SEED}, population={POPULATION_SIZE}, "
          f"generations={GENERATIONS}, mutation rate={MUTATION_RATE}, "
          f"elites={ELITE_COUNT}\n")
    route, distance = run_ga()
    assert is_valid(route)
    print("\n=== RESULT ===")
    print("Best route    :", format_route(route))
    print("Total distance:", distance, "km")