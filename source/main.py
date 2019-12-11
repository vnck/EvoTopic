from Documents import Documents
from GA import GA
import pandas as pd

data_path = '../data/github_issues.csv'

df = pd.read_csv(data_path)

docs = Documents()
docs.load(list(df['description'])[:50])
docs.vectorise()

print(docs.get_doc_size())

GA = GA(docs.get_vectors(), docs.get_dictionary(),pop_size=10,fitness_budget=10000)
GA.initialise_population()
GA.evolve()

# fittest_individual = GA.get_fittest()
# fittest_individual.evaluate()