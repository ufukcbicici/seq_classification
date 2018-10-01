from corpus.symbolic_corpus import SymbolicCorpus
from global_constants import GlobalConstants

corpus = SymbolicCorpus()
corpus.read_documents(path=GlobalConstants.LARGE_TRAINING_SET, is_training=True)
corpus.read_documents(path=GlobalConstants.LARGE_TEST_SET, is_training=False)
print("X")