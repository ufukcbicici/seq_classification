from auxilliary.db_logger import DbLogger
from classifiers.rnn_classifier import RnnClassifier
from classifiers.svm_classifier import SvmClassifier
from corpus.symbolic_corpus import SymbolicCorpus
from embedding_training.cbow_embedding_generator import CbowEmbeddingGenerator
from embedding_training.context_generator import ContextGenerator
from global_constants import GlobalConstants, DatasetType


# Tf-Idf analysis
# def tf_idf_analysis(validation_ratio=0.1):
#


def large_training_set_pipeline(
        create_vocabulary_from_scratch, extract_embeddings, validation_ratio=0.1):
    # Corpus Creation
    corpus = SymbolicCorpus()
    corpus.read_documents(path=GlobalConstants.LARGE_TRAINING_SET, is_training=True)
    corpus.read_documents(path=GlobalConstants.LARGE_TEST_SET, is_training=False)
    corpus.pick_validation_set(validation_ratio=validation_ratio)
    if create_vocabulary_from_scratch:
        corpus.analyze_data()
    else:
        corpus.read_vocabulary()

    # Embedding Training
    if extract_embeddings:
        ContextGenerator.build_cbow_tokens(corpus=corpus)
        context_generator = ContextGenerator()
        context_generator.read_cbow_data()
        cbow_embedding = CbowEmbeddingGenerator(corpus=corpus, context_generator=context_generator)
        cbow_embedding.train()

    # RNN Model Training
    corpus.split_dataset_into_sequences()
    rnn_classifier = RnnClassifier(corpus=corpus)
    rnn_classifier.build_classifier()
    rnn_classifier.train()


def small_training_set_pipeline(validation_ratio=0.1):
    corpus = SymbolicCorpus()
    corpus.read_documents(path=GlobalConstants.SMALL_TRAINING_SET, is_training=True)
    corpus.read_documents(path=GlobalConstants.SMALL_TEST_SET, is_training=False)
    corpus.pick_validation_set(validation_ratio=validation_ratio)
    corpus.tf_idf_analysis()
    # SVM classifier
    svm_classifier = SvmClassifier(corpus=corpus)
    svm_classifier.train()
    svm_classifier.test(dataset_type=DatasetType.Training)
    svm_classifier.test(dataset_type=DatasetType.Validation)


# large_training_set_pipeline(create_vocabulary_from_scratch=False, extract_embeddings=False)
small_training_set_pipeline()
