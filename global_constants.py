import os


class GlobalConstants:
    THREAD_COUNT = 1
    CORPUS_FREQUENCY_THRESHOLD = 3
    MAX_CLUSTER_FREQ_RATIO = 0.25
    CBOW_WINDOW_SIZE = 2
    # CBOW Training
    EMBEDDING_BATCH_SIZE = 128
    EMBEDDING_SIZE = 128
    NUM_NEGATIVE_SAMPLES = 128
    EPOCH_COUNT = 10
    VALIDATION_TOKENS = {"Z4",    "Z15",    "X344420902",    "Y3143",    "Y1222",    "Y1012203",    "X264648477",
                         "X212549706",    "Y1040796",    "Y3337",    "Y3331",    "Y3363",    "Z2",    "Z3",    "Z7",
                         "Z36",    "Z33",    "Z8"}
    # PATHS
    # Idea GPU
    LARGE_TRAINING_SET = os.path.join("D:\\", "deep", "seq_classification", "data", "training-data-large.txt")
    LARGE_TEST_SET = os.path.join("D:\\", "deep", "seq_classification", "data", "test-data-large.txt")
    EMBEDDING_CHECKPOINT_PATH = os.path.join("D:\\", "deep", "seq_classification", "embedding_training",
                                             "embeddings")
    # Home
    # LARGE_TRAINING_SET = os.path.join("D:\\", "seq_classification", "data", "training-data-large.txt")
    # LARGE_TEST_SET = os.path.join("D:\\", "seq_classification", "data", "test-data-large.txt")
    # EMBEDDING_CHECKPOINT_PATH




