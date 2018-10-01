from corpus.symbolic_corpus import SymbolicCorpus
from global_constants import GlobalConstants

corpus = SymbolicCorpus()
corpus.read_documents(path=GlobalConstants.LARGE_TRAINING_SET)
print("X")