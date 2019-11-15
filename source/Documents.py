import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Phrases
from gensim.corpora import Dictionary

class Documents:
  """
  A class to store documents and document vectors.

  Attributes
  ----------
  docs : dataframe
    a dataframe containing text and vectors

  doc_size : int
    number of docs in corpus

  vocab_size : int
    size of vocabulary of corpus
  

  Methods
  -------
  load(path)
    reads files and stores text data in a dataframe

  vectorise()
    converts textual format to vectorised format

  get_raw()
    returns list of documents in raw textual format

  get_tokens()
    returns list of documents in tokenised format

  get_vectors()
    returns list of documents in vectorised format

  get_doc_size()
    returns number of documents

  get_vocab_size()
    returns number of unique words in corpus

  get_vocab()
    returns list of corpus vocabulary
  """

  def __init__(self):
    self.df = pd.DataFrame()

    self.tokenizer = RegexpTokenizer(r'\w+')
    self.lemmatizer = WordNetLemmatizer()
    self.stopwords = set(stopwords.words('english'))

  def load(self, path):
    self.df = self.df.append(pd.DataFrame({'raw':['hello world time','mom dad boy','tiger lions bear','hawk eagle 23 penguin']}))

  def __preprocessing_pipeline(self, doc):
    doc = doc.lower()
    tokens = self.tokenizer.tokenize(doc)
    tokens = [t for t in tokens if not t.isnumeric()]
    tokens = [t for t in tokens if t not in self.stopwords]
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
    return tokens

  def __add_bigrams(self, doc):
    for token in self.bigrams[doc]:
      if '_' in token:
        doc.append(token)
    return doc

  def vectorise(self):
    self.df['tokens'] = self.df['raw'].apply(self.__preprocessing_pipeline)
    self.bigrams = Phrases(self.get_tokens(), min_count=20)
    self.df['tokens'] = self.df['tokens'].apply(self.__add_bigrams)
    self.dictionary = Dictionary(self.get_tokens())
    # self.dictionary.filter_extremes(no_below=20, no_above=0.5)
    self.df['vectors'] = self.df['tokens'].apply(self.dictionary.doc2bow)

  def get_raw(self):
    return list(self.df['raw'])
  
  def get_vectors(self):
    return list(self.df['vectors'])

  def get_tokens(self):
    return list(self.df['tokens'])

  def get_doc_size(self):
    return len(self.dictionary.num_docs)

  def get_vocab_size(self):
    return len(self.dictionary.num_nnz)

  def get_vocab(self):
    return list(self.dictionary.values())