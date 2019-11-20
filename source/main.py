from documents import Documents
from ga import GA
import pandas as pd

data_path = '../data/github_issues.csv'

df = pd.read_csv(data_path)

docs = Documents()
docs.load(list(df['description']))
docs.vectorise()

print(docs.get_doc_size())
print(docs.get_vectors())

# GA = GA(docs,pop_size=100,fitness_budget=10000)
# GA.initialise_population()
# GA.evolve()

# fittest_individual = GA.get_fittest()
# fittest_individual.evaluate()