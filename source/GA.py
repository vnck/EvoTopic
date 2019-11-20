from gene import Gene

class GA:
  """
  A class used to contain our genetic algorithm.

  Attributes
  ----------
  docs : Document
    Document class object containing document to evaluate fitness on.
  
  population : list
    list of Gene class objects to perform selection, crossover, and mutation on.
  
  population_size : int
    size of the population.
  
  fitness_budget : float
    the number of evaluations remaining before the GA halts.
  
  fittest : Gene
    the fittest Gene class object.
  
  fitness : float
    fitness score of the best Gene class object.

  Methods
  -------
  initialise_population()
    randomly generates an initial population.

  evolve()
    calls selection, crossover, and mutation loop while fitness budget > 0.
  
  selection()
    performs selection on the current population.

  crossover()
    performs crossover between two Gene objects in the population with a probability.

  mutate()
    calls the mutate method of each Gene class given a probability.

  calculate_fitness(gene)
    calculates the fitness of a gene.

  update_population_fitness()
    updates fitness of all genes in the new population and updates the fittest individual.

  get_fittest()
    returns the fittest gene.

  """

  def __init__(self,docs,pop_size=100,fitness_budget=10000):
    self.docs = docs
    self.population = []
    self.population_size = pop_size
    self.fitness_budget = fitness_budget
    self.fittest = None
    self.fitness = 0.0

  def initialise_population(self):
    # TODO: random initialisation of population
    pass

  def evolve(self):
    while(self.fitness_budget > 0):
      self.selection()
      self.crossover()
      self.mutate()
      self.update_population_fitness()
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

  def update_population_fitness(self):
    # TODO: calls calculate_fitness on all genes in the new population and updates the fittest individual
    pass

  def calculate_fitness(self,gene):
    # TODO: calculate silhouette value and update fitness of gene
    pass

  def get_fittest(self):
    return self.fittest