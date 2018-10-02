from auxilliary.db_logger import DbLogger
from corpus.symbolic_corpus import SymbolicCorpus
from global_constants import GlobalConstants

corpus = SymbolicCorpus()
corpus.read_documents(path=GlobalConstants.LARGE_TRAINING_SET, is_training=True)
corpus.read_documents(path=GlobalConstants.LARGE_TEST_SET, is_training=False)
corpus.pick_validation_set(validation_ratio=0.1)
corpus.analyze_data()
print("X")