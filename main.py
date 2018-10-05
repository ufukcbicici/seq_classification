from auxilliary.db_logger import DbLogger
from corpus.symbolic_corpus import SymbolicCorpus
from embedding_training.cbow_embedding_generator import CbowEmbeddingGenerator
from embedding_training.context_generator import ContextGenerator
from global_constants import GlobalConstants


# Corpus-Vocabulary Creation
def create_corpus(create_from_scratch=True, validation_ratio=0.1):
    corpus = SymbolicCorpus()
    corpus.read_documents(path=GlobalConstants.LARGE_TRAINING_SET, is_training=True)
    corpus.read_documents(path=GlobalConstants.LARGE_TEST_SET, is_training=False)
    corpus.pick_validation_set(validation_ratio=validation_ratio)
    if create_from_scratch:
        corpus.analyze_data()
    else:
        corpus.read_vocabulary()
    return corpus


# Create token embeddings
def create_embeddings(corpus, create_from_scratch=True):
    if create_from_scratch:
        ContextGenerator.build_cbow_tokens(corpus=corpus)
    context_generator = ContextGenerator()
    context_generator.read_cbow_data()
    cbow_embedding = CbowEmbeddingGenerator(corpus=corpus, context_generator=context_generator)
    cbow_embedding.train()


# Pipeline
corpus = create_corpus(create_from_scratch=False)
create_embeddings(corpus=corpus, create_from_scratch=False)


