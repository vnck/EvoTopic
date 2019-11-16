import source.Documents
import source.GA

data_path = ''

docs = Documents()
docs.load(data_path)
docs.vectorise()

GA = GA(docs,pop_size=100,fitness_budget=10000)
GA.initialise_population()
GA.evolve()

fittest_individual = GA.get_fittest()
fittest_individual.evaluate()