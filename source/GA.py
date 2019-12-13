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
import pprint
import numpy as np
import pyLDAvis.gensim


MUTATION_RATIO = 0.3
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
    self.corpus = docs
    self.docs_size = len(self.corpus)
    self.dictionary = dictionary
    self.vocab_size = len(self.dictionary)
    Gene.set_vocab_size(self.vocab_size)
    Gene.set_doc_size(self.docs_size)
    self.population = []
    self.population_size = pop_size
    self.fitness_budget = fitness_budget
    self.fitness = -999.0
    self.bestGene = Gene()
    self.iteration = 0

  def initialise_population(self):
    """Random initialisation of population"""
    print('Initialising Population...')
    self.population = [Gene() for i in range(self.population_size)]
    self.update_population_fitness()
    # print('{}: Fitness: {} Fitness Budget: {} '.format(self.iteration,self.fitness,self.fitness_budget))
    # print('{}: Gene.n: {} Gene.a: {} Gene.b: {} '.format(self.iteration, self.bestGene.n, len(self.bestGene.a), len(self.bestGene.b)))
  
  def evolve(self):
    print('Evolving Population...')
    while(self.fitness_budget > 0):
      self.selection()
      self.crossover()
      self.mutate()
      self.update_population_fitness()
      self.iteration += 1
      self.fitness_budget -= 1
      # print('{}: Gene.n: {} Gene.a: {} Gene.b: {} '.format(self.iteration, self.bestGene.n, len(self.bestGene.a), len(self.bestGene.b)))
      if round(self.fitness,15) == 0:
        break
  
  def selection(self):
    """Top 20% of population will be selected"""
    # Sort population
    self.population = sorted(self.population, key=lambda gene: gene.fitness, reverse=True)
    # Get top 20% from population
    self.population = self.population[:int(self.population_size*SELECT_RATIO)]
  

  def __crossover2genes(self, gene1, gene2):
    """Crossover two genes"""
    new_gene_n = 0
    new_gene_a = []
    new_gene_b = []
    # Which part do you want to crossover?  
    crossover_part = random.choice(["n", "b"])
    if crossover_part == "n":
      # Average of two genes
      new_gene_n = math.ceil((gene1.n+gene2.n)/2)
      for i in range(new_gene_n):
        if ((len(gene1.a)-1) < i):
          new_gene_a.append(gene2.a[i])
        elif ((len(gene2.a)-1) < i):
          new_gene_a.append(gene1.a[i])
        else :
          new_gene_a.append((gene1.a[i]+gene2.a[i])/2)
      new_gene_b = gene1.b[:]
    else:
      # Average of two genes
      for i in range(self.vocab_size):
        new_gene_b.append((gene1.b[i]+gene2.b[i])/2)
      new_gene_n = gene1.n
      new_gene_a = gene1.a[:]
    # normalization
    new_gene_a = [float(i)/sum(new_gene_a) for i in new_gene_a]
    new_gene_b = [float(i)/sum(new_gene_b) for i in new_gene_b]
    new_gene = Gene(new_gene_n, new_gene_a, new_gene_b)
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
<<<<<<< HEAD
    # for p in self.population:
      # if (0 in p.a):
      #   print("Before Mutation: Zero in p.a")
      # if (0 in p.b):
      #   print("Before mutation: Zero in p.b")
    new_population = [p.mutate(MUTATION_RATIO) for p in self.population]
    for p in new_population:
      if (0 in p.a):
        # print("After Mutation: Zero in p.a")
        while(0 in p.a):
          p.a[p.a.index(0)] = 0.01
      if (0 in p.b):
        # print("After mutation: Zero in p.b")
        while(0 in p.b):
          p.b[p.b.index(0)] = 0.01
=======
    new_population = []
    for p in self.population:
      new_p = p.mutate(MUTATION_RATIO)
      if (0 in p.a):
        print("After Mutation: Zero in p.a")
      if (0 in p.b):
        print("After Mutation: Zero in p.b")
      new_population.append(new_p)
>>>>>>> 737a1ec9493e2f434ecf8774ab9384fee7ab535b
    self.population = new_population

  def update_population_fitness(self):
    # calls calculate_fitness on all genes in the new population and updates the fittest individual
    pop_fitness = 0.0
    for p in tqdm(self.population):
      # p.fitness = abs(self.calculate_fitness(p))
      p.fitness = self.calculate_fitness(p)
      # Update best fitness
      # if p.fitness < self.fitness:
      if p.fitness > self.fitness:
        pop_fitness = p.fitness
        self.bestGene = p
    self.fitness = pop_fitness
    print('{}: Fitness: {:.15f}, Best Fitness: {:.15f}, Num Topics: {}, Fitness Budget: {} '.format(self.iteration,pop_fitness,self.fitness,self.bestGene.n,self.fitness_budget))

  def calculate_fitness(self,gene):
    # Make LDA model 
    # assert gene.n == len(gene.a), "n: {}, a:{}".format(gene.n, len(gene.a))
    # assert len(gene.b) == self.vocab_size, "b: {}, v:{}".format(len(gene.b), self.vocab_size)
    lda = LdaModel(corpus = self.corpus,
                   id2word = self.dictionary,
                   num_topics = gene.n,
                   alpha = gene.a,
                   eta = gene.b)
    # Classify docs using LDA model
    cm = CoherenceModel(model=lda, corpus=self.corpus, coherence='u_mass')
    # Calculate silhouette score
    result = cm.get_coherence()
    return result
<<<<<<< HEAD

  # def calculate_fitness(self,gene):
  #   # Make LDA model 
  #   lda = LdaModel(corpus = self.corpus,
  #                  id2word = self.dictionary,
  #                  num_topics = gene.n,
  #                  alpha = gene.a,
  #                  eta = gene.b)
  #   # Classify docs using LDA model
  #   labels = []
  #   word_cntLst = []
  #   if(len(self.corpus)<2):
  #     return -1
  #   for text in self.corpus:
  #     # Make label list
  #     topic_probLst = lda.get_document_topics(text)
  #     if (len(topic_probLst) == 0):
  #       # print("LDA is fucked")
  #       print("sum of Eta: ", sum(gene.b))
  #       print("sum of Alpha: ", sum(gene.a))
  #       return -1
  #     labels.append(max(topic_probLst, key=lambda tup: tup[1])[0])
  #     # Make word count list
  #     words = [0]*self.vocab_size
  #     for tup in text:
  #       words[tup[0]] = tup[1]
  #     word_cntLst.append(words[:])
  #   # Calculate silhouette score
  #   if(len(np.unique(labels)) < 2):
  #     return -1
  #   return metrics.silhouette_score(word_cntLst, labels, metric='cosine')
=======
  '''
  def calculate_fitness(self,gene):
    # Make LDA model 
    lda = LdaModel(corpus = self.corpus,
                   num_topics = gene.n,
                   alpha = gene.a,
                   eta = gene.b)
    # Classify docs using LDA model
    labels = []
    word_cntLst = []
    if(len(self.corpus)<2):
      return -1
    for text in self.corpus:
      # Make label list
      topic_probLst = lda.get_document_topics(text)
      if (len(topic_probLst) == 0):
        print("LDA is fucked")
        if (0 in gene.b) :
          print("calculate fitness: Zero in b")
        return -1
      labels.append(max(topic_probLst, key=lambda tup: tup[1])[0])
      # Make word count list
      words = [0]*self.vocab_size
      for tup in text:
        words[tup[0]] = tup[1]
      word_cntLst.append(words[:])
    # Calculate silhouette score
    if(len(np.unique(labels)) < 2):
      return -1
    return metrics.silhouette_score(word_cntLst, labels, metric='cosine')
>>>>>>> 737a1ec9493e2f434ecf8774ab9384fee7ab535b

  def get_fittest(self):
    return self.bestGene

  def get_model(self):
    lda = LdaModel(corpus = self.corpus,
                   id2word = self.dictionary,
                   num_topics = self.bestGene.n,
                   alpha = self.bestGene.a,
                   eta = self.bestGene.b)
    return lda
