import source.Gene

class GA:
  def __init__(self,pop_size=100,fitness_budget=10000):
    self.population = []
    self.population_size = pop_size
    self.fitness_budget = fitness_budget
    self.fittest = None
    self.fitness = 0.0

  def initialise_population(self):
    # TODO: random initialisation of population
    pass

  def evolve(self):
    # TODO: begin GA process until fitness budget < 0
    pass
  
  def selection(self):
    # TODO: selection from current population to populate new population
    pass
  
  def crossover(self):
    # TODO: crossover operator
    pass
  
  def mutate(self):
    # TODO: calls mutate() for each gene in population
    pass

  def get_fittest(self):
    return self.fittest