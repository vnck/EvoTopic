import random
import numpy as np

class Gene:
  """
  A class used to represent an individual in the population. It contains the hyperparameters for LDA algorithm.

  Attributes
  ----------
  n : int
    the number of topic clusters to be generated by LDA

  a : list
    1D array of length equal to number of topic clusters (n), that expresses our a-priori belief for the each topics’ probability.

  b : list
    1D array of length equal to vocab_size, that expresses our a-priori belief for each word.

  fitness : float
    fitness score of the gene

  vocab_size : int
    the size of the vocabulary of corpus.

  N_MIN : int
    minimum value of n

  N_MAX : int
    maximum value of n

  Methods
  -------
  set_vocab_size(vocab_size)
    static method that sets the value of static variable vocab_size.

  mutate()
    mutates values of n, a, and b.

  set_fitness()
    sets the fitness score of a gene.

  get_fitness()
    returns the fitness score of a gene.

  """

  n = 1
  N_MIN = 1
  N_MAX = 1
  a = []
  b = []
  vocab_size = 0
  fitness = -1

  def __init__(self, n=None, a=None, b=None):
    """
    Parameters
    ----------
    n : int, optional
      the number of topic clusters to be generated by LDA. Default value is a
      randomly generated integer between N_MIN and N_MAX.

    a : list, optional
      1D array of length equal to number of topic clusters (n), that expresses
      our a-priori belief for the each topics’ probability. Default value is a 
      randomly generated 1D array.

    b : list, optional
      1D array of length equal to vocab_size, that expresses our a-priori belief
      for each word. Default value is a randomly generated 1D array.
    """

    if Gene.vocab_size < 1 or not isinstance(Gene.vocab_size,int):
      raise ValueError('vocab_size should be a positive integer. Set vocab_size using set_vocab_size method. The value of vocab_size was: {}'.format(Gene.vocab_size))
      
    if n is None or a is None or b is None:
      self.n = np.random.randint(self.N_MIN,self.N_MAX)
      self.a = np.random.dirichlet(np.ones(self.n), size=1)[0]
      self.b = np.random.dirichlet(np.ones(Gene.vocab_size), size=1)[0]
    else:
      if not isinstance(n, int):
        raise Exception('n should be a positive integer. \
                         The value of n was: {}'.format(n))
      self.n = n
      self.a = a
      self.b = b

  @staticmethod
  def set_vocab_size(vocab_size):
    """Sets the value of vocab_size. Must be set to a positive integer before a Gene instance can be created.

    Parameters
    ----------
    vocab_size : int
      size of vocabulary of corpus.
    """
    Gene.vocab_size = vocab_size

  @staticmethod
  def set_doc_size(doc_size):
    """Sets the value of doc_size. Must be set to a positive integer before a Gene instance can be created.

    Parameters
    ----------
    doc_size : int
      size of documents of corpus.
    """
    Gene.N_MAX = doc_size

  def mutate(self, mutation_rate):
    if(random.random() < mutation_rate):
      """ mutate n """
      self.n = random.randint(1, self.N_MAX)

      """ mutate a """
      # Randomly sample probabilities of topics in a
      genes_a = random.sample(self.a, random.randint(2, len(self.a)))
      
      # Calculate the sum of probabilities of topics sampled
      sum_p_a = 0
      for i in genes_a:
        sum_p_a += self.a[genes_a.index(i)]
      
      # Redistribute the probabilities among the topics sampled
      leftover_p_a = sum_p_a
      count_a = 0
      for i in genes_a:
        if count_a == len(genes_a) - 1:
          # Assign the leftover probability to the last one sampled
          self.a[genes_a.index(i)] = leftover_p_a
        else:  
          # Generate random float between 0 and 1
          ra = random.random()
          # Assign the value in range of sum_p_a to the ith probability of topic sampled
          self.a[genes_a.index(i)] = leftover_p_a * ra
          # Update leftover_p_a
          leftover_p_a -= self.a[genes_a.index(i)]
        count_a += 1

      """ mutate b """
      # Randomly sample probabilities of words in b
      genes_b = random.sample(self.b, random.randint(2, len(self.b)))
      
      # Calculate the sum of sampled probabilities of words
      sum_p_b = 0
      for i in genes_b:
        sum_p_b += self.b[genes_b.index(i)]
      
      # Redistribute the probabilities among the sampled probabilities of words 
      leftover_p_b = sum_p_b
      count_b = 0
      for i in genes_b:
        if count_b == len(genes_b) - 1:
          # Assign the leftover probability to the last one sampled
          self.b[genes_b.index(i)] = leftover_p_b
        else:  
          # Generate random float between 0 and 1
          rb = random.random()
          # Assign the value in range of sum_p_a to the ith probability of word sampled
          self.b[genes_b.index(i)] = leftover_p_b * rb
          # Update leftover_p_b
          leftover_p_b -= self.b[genes_b.index(i)]
        count_b += 1


  def set_fitness(self,f):
    self.fitness = f
