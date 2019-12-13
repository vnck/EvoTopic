from Documents import Documents
from GA import GA
import pandas as pd

data_path = '../data/github_issues.csv'

df = pd.read_csv(data_path)

docs = Documents()
docs.load(list(df['description'])[:100])
docs.vectorise()

print("No. of documents loaded: {}".format(docs.get_doc_size()))

GA = GA(docs.get_vectors(), docs.get_dictionary(),pop_size=20,fitness_budget=10000)
GA.initialise_population()
GA.evolve()
model = GA.get_model()

# fittest_individual = GA.get_fittest()
