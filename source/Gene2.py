import random
import numpy as np
import copy

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
      self.a = np.random.dirichlet(np.ones(self.n), size=1)[0].tolist()
      self.b = np.random.dirichlet(np.ones(Gene.vocab_size), size=1)[0].tolist()
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

  def partition_float(self, a, n):
    assert a > 0, "Gene.py partition_float: a should be positive number a= {}".format(a)
    if n == 1:
      return [a]
    pieces = []
    for i in range(n):
      # pieces.append(random.uniform(0.0001,a-sum(pieces)-0.0001))
      pieces.append(random.uniform(0.0001, 1))
    # pieces.append(a-sum(pieces))
    # normalize pieces
    pieces = [float(i)/sum(pieces) for i in pieces]
    # set sum of pieces to a
    for i in range(n):
      pieces[i] = pieces[i]*a
    return pieces

  def mutate(self, mutation_rate):
    if(random.random() < mutation_rate):
      """ mutate n """
      self.n = random.randint(1, self.N_MAX)
    
    """ then mutate a """
    if len(self.a) > self.n:
      # print('n:{} < a:{}'.format(self.n, len(self.a)))
      # randomly drop probabilities
      n_diff = len(self.a) - self.n
      leftover_prob = 0.0
      for i in range(n_diff):
        leftover_prob += self.a.pop(random.randrange(len(self.a)))
      # print(len(self.a))
      # randomly add probabilities until sum to 1
      n_distribute = random.randrange(len(self.a))
      spare_prob = self.partition_float(leftover_prob, n_distribute)
      for p in spare_prob:
        idx = random.randrange(len(self.a))
        self.a[idx] += p
      # print("Gene.py case a > n : sum of self.a = ", sum(self.a))

    elif len(self.a) < self.n:
      # print('n:{} > a:{}'.format(self.n, len(self.a)))
      # randomly remove probabilities from original
      n_diff = self.n - len(self.a)
      remove_portion = random.random()
      leftover_prob = 0
      for i in range(len(self.a)):
        leftover_prob += self.a[i]*remove_portion
        self.a[i] = self.a[i]*(1-remove_portion)
      # redistribute leftover_prob
      spare_prob = self.partition_float(leftover_prob, n_diff)
      # append self.a
      self.a += spare_prob
      '''
      # randomly add probabilities
      n_diff = self.n - len(self.a)
      for i in range(n_diff):
        # self.a.insert(random.randrange(len(self.a)), random.random())
        self.a.insert(random.randrange(len(self.a)), random.uniform(0.00000001, 0.99999999))
      # randomly remove probabilities until sum to 1
      n_distribute = random.randrange(len(self.a))
      spare_prob = self.partition_float(sum(self.a) - 1, n_distribute)
      for p in spare_prob:
        idx = random.randrange(len(self.a))
        while(self.a[idx] - p <= 0):
          idx = random.randrange(len(self.a))
        self.a[idx] -= p
      '''
      # print("Gene.py case a < n : sum of self.a = ", sum(self.a))
        

    elif (random.random() < mutation_rate):
      """ maybe mutate a if n does not change """
      # print('n:{} == a:{}'.format(self.n, len(self.a)))
      if len(self.a) != 1:
        n_choice = random.sample([i for i in range(len(self.a))], random.randrange(1,len(self.a)))
        leftover_prob = 0.0
        for i in sorted(n_choice, reverse = True):
          leftover_prob += self.a.pop(i)
        spare_prob = self.partition_float(1 - sum(self.a), len(n_choice))
        for p in spare_prob:
          self.a.insert(n_choice.pop(random.randrange(len(n_choice))),p)
      else:
        print ("No mutation since a has only one element!")

      # # Randomly sample probabilities of topics in a
      # genes_a = random.sample(self.a, random.randint(2, len(self.n)))
      
      # # Calculate the sum of probabilities of topics sampled
      # sum_p_a = sum([self.a[genes_a.index(i)] for i in genes_a])
      
      # # Redistribute the probabilities among the topics sampled
      # leftover_p_a = sum_p_a
      # count_a = 0
      # for i in genes_a:
      #   if count_a == len(genes_a) - 1:
      #     # Assign the leftover probability to the last one sampled
      #     self.a[genes_a.index(i)] = leftover_p_a
      #   else:  
      #     # Generate random float between 0 and 1
      #     ra = random.random()
      #     # Assign the value in range of sum_p_a to the ith probability of topic sampled
      #     self.a[genes_a.index(i)] = leftover_p_a * ra
      #     # Update leftover_p_a
      #     leftover_p_a -= self.a[genes_a.index(i)]
      #   count_a += 1

    if(random.random() < mutation_rate):
      """ maybe mutate b """
      # Randomly sample probabilities of words in b
      genes_b = random.sample(self.b, random.randint(2, len(self.b)))
      # Calculate the sum of sampled probabilities of words
      sum_p_b = sum(genes_b)
      # Redistribute the probabilities among the sampled probabilities of words 
      distribute_list = []
      for i in range(len(genes_b)):
        distribute_list.append(random.random())
      distribute_list = [float(i)/sum(distribute_list) for i in distribute_list]
      for i in range(len(genes_b)):
        self.b[self.b.index(genes_b[i])] = distribute_list[i]*sum_p_b
    
    assert self.n == len(self.a), "n: {}, a:{}".format(self.n, len(self.a))
    assert len(self.b) == self.vocab_size, "b: {}, v:{}".format(len(self.b), self.vocab_size)
    new_gene = copy.deepcopy(self)
    return new_gene

  def set_fitness(self,f):
    self.fitness = f