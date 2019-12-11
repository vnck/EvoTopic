from Gene import Gene
import random 
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import gensim
from tqdm import tqdm
from gensim.models.coherencemodel import CoherenceModel
import math


MUTATION_RATIO = 0.1
SELECT_RATIO = 0.2

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

  def __init__(self,docs,dictionary,pop_size=100,fitness_budget=10000):
    # initial setting
    if (pop_size <= 5):
      print('Population size must be at least > 5, setting population size to 10.')
      pop_size = 10
    self.corpus = docs
    self.docs_size = len(self.corpus)
    self.dictionary = dictionary
    self.vocab_size = len(self.dictionary)
    self.population = []
    self.population_size = pop_size
    self.fitness_budget = fitness_budget
    self.fittest = self.calculate_fitness
    self.fitness = -1.0
    self.bestGene = None
    Gene.set_vocab_size(self.vocab_size)
    Gene.set_doc_size(self.docs_size)
    self.iteration = 0

  def initialise_population(self):
    """Random initialisation of population"""
    print('Initialising Population...')
    self.population = [Gene() for i in range(self.population_size)]
    self.update_population_fitness()
    print('{}: Fitness: {} Fitness Budget: {} '.format(self.iteration,self.fitness,self.fitness_budget))

  def evolve(self):
    print('Evolving Poppulation...')
    while(self.fitness_budget > 0):
      self.selection()
      self.crossover()
      self.mutate()
      self.update_population_fitness()
      self.iteration += 1
      self.fitness_budget -= 1
      print('{}: Fitness: {} Fitness Budget: {} '.format(self.iteration,self.fitness,self.fitness_budget))
  
  def selection(self):
    """Top 20% of population will be selected"""
    # Sort population
    self.population = sorted(self.population, key=lambda gene: gene.fitness, reverse=True)
    # Get top 20% from population
    self.population = self.population[:int(self.population_size*SELECT_RATIO)]
  

  def __crossover2genes(self, gene1, gene2):
    """Crossover two genes"""
    new_gene = Gene()
    # Flip coin to decide victim
    coin = random.choice([True, False])
    if coin:
      new_gene.n = gene1.n
      new_gene.a = gene1.a
      new_gene.b = gene1.b
    else:
      new_gene.n = gene2.n
      new_gene.a = gene2.a
      new_gene.b = gene2.b
    # Which part do you want to crossover?  
    crossover_part = random.choice(["n", "a", "b"])
    if crossover_part == "n":
      # Average of two genes
      new_gene.n = math.ceil((gene1.n+gene2.n)/2)
    elif crossover_part == "a":
      # Choose random point 
      crossover_point = random.randint(0, len(gene1.a)-1)
      new_gene.a = gene1.a[:crossover_point]+gene2.a[crossover_point:]
    else:
      # Choose random point 
      crossover_point = random.randint(0, len(gene2.b)-1)
      new_gene.b = gene1.b[:crossover_point]+gene2.b[crossover_point:]
    return new_gene

  def crossover(self):
    """Generate new population using crossover"""
    while(len(self.population) < self.population_size):
      #Randomly select two genes
      # gene1, gene2 = random.sample(self.population[:int(self.population_size*SELECT_RATIO)], 2)
      gene1, gene2 = random.sample(self.population, 2)
      new_gene = self.__crossover2genes(gene1, gene2)
      self.population.append(new_gene)

  def mutate(self):
    new_population = [p.mutate(MUTATION_RATIO) for p in self.population]
    self.population = new_population

  def update_population_fitness(self):
    # TODO: calls calculate_fitness on all genes in the new population and updates the fittest individual
    for p in tqdm(self.population):
      p.fitness = self.calculate_fitness(p)
      # Update best fitness
      if p.fitness > self.fitness:
        self.fitness = p.fitness
        self.bestGene = p

  def calculate_fitness(self,gene):
    # Make LDA model 
    assert gene.n == len(gene.a), "n: {}, a:{}".format(gene.n, len(gene.a))
    assert len(gene.b) == self.vocab_size, "b: {}, v:{}".format(len(gene.b), self.vocab_size)
    lda = LdaModel(corpus = self.corpus,
                   id2word = self.dictionary,
                   num_topics = gene.n,
                   alpha = gene.a,
                   eta = gene.b)
    # Classify docs using LDA model
    cm = CoherenceModel(model=lda, corpus=self.corpus, coherence='u_mass')
    # Calculate silhouette score 
    return cm.get_coherence()

  # def calculate_fitness(self,gene):
  #   # Make LDA model 
  #   lda = LdaModel(corpus = self.corpus,
  #                  num_topics = gene.n,
  #                  alpha = gene.a,
  #                  eta = gene.b)
  #   # Classify docs using LDA model
  #   labels = []
  #   word_cntLst = []
  #   for text in self.corpus:
  #     # Make label list
  #     topic_probLst = lda.get_document_topics(text)
  #     if (len(topic_probLst) == 0):
  #       print(text)
  #     labels.append(max(topic_probLst, key=lambda tup: tup[1])[0])
  #     # Make word count list
  #     words = [0]*self.vocab_size
  #     for tup in text:
  #       words[tup[0]] = tup[1]
  #     word_cntLst.append(words[:])
  #   # Calculate silhouette score 
  #   return metrics.silhouette_score(word_cntLst, labels, metric='cosine')

  def get_fittest(self):
    return self.fittest
