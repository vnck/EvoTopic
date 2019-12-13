from Documents import Documents
from GA import GA
import pandas as pd
import pickle
from os import path

loadit = False

if loadit:
  docs = pickle.load(open('docs.pkl', 'rb'))
else:
  data_path = '../data/github_issues.csv'
  df = pd.read_csv(data_path)
  docs = Documents()
#   docs.load(list(df['description'])[:300])
  docs.load(list(df['description']))
  docs.vectorise()
  pickle.dump(docs, open('docs.pkl', 'wb+'))

print("No. of documents loaded: {}".format(docs.get_doc_size()))

corpus = docs.get_vectors()
dictionary = docs.get_dictionary()

GA = GA(corpus,dictionary,pop_size=30,fitness_budget=10000,objective='coherence')
GA.initialise_population()
GA.evolve()

fittest_gene = GA.get_fittest()
# model = GA.get_model()
# docs.assign_labels(model)

print('Fittest Gene discovered {} topics with score of {:.8f}'.format(fittest_gene.n,GA.fitness))