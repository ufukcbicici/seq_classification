import os
from enum import IntEnum


class DatasetType(IntEnum):
    Training = 0
    Validation = 1
    Test = 2


class GlobalConstants:
    THREAD_COUNT = 1
    CORPUS_FREQUENCY_THRESHOLD = 3
    MAX_CLUSTER_FREQ_RATIO = 0.25
    CBOW_WINDOW_SIZE = 2
    # CBOW Training
    EMBEDDING_BATCH_SIZE = 128
    EMBEDDING_SIZE = 128
    NUM_NEGATIVE_SAMPLES = 128
    EPOCH_COUNT = 25
    VALIDATION_TOKENS = {"Z4", "Z15", "X344420902", "Y3143", "Y1222", "Y1012203", "X264648477",
                         "X212549706", "Y1040796", "Y3337", "Y3331", "Y3363", "Z2", "Z3", "Z7",
                         "Z36", "Z33", "Z8"}
    # RNN Training
    FINE_TUNE_EMBEDDINGS = True
    L2_LAMBDA_COEFFICENT = 0.0
    INITIAL_LR_CLASSIFIER = 0.01
    BATCH_SIZE = 128
    USE_INPUT_DROPOUT = True
    NUM_OF_LSTM_LAYERS = 1
    USE_BIDIRECTIONAL_LSTM = True
    LSTM_HIDDEN_LAYER_SIZE = 128
    USE_ATTENTION_MECHANISM = True
    DROPOUT_KEEP_PROB = 1.0
    EPOCH_COUNT_CLASSIFIER = 20
    MAX_SEQUENCE_LENGTH = 250
    SEQUENCE_SLIDING_WINDOW_SIZE = 250

    # CNN Training
    CNN_SLIDING_WINDOWS = [i + 1 for i in range(int(MAX_SEQUENCE_LENGTH / 2))]

    # Classical Feature Extraction
    N_GRAMS = {1}

    # PATHS
    # Idea GPU
    # ROOT_PATH = os.path.join("D:\\", "deep", "seq_classification")
    # Home
    # ROOT_PATH = os.path.join("D:\\", "seq_classification")
    # Idea
    # ROOT_PATH = os.path.join("C:\\", "Users", "ufuk.bicici", "Desktop", "seq_classification")
    # DGX
    ROOT_PATH = os.path.join("/raid", "users", "ucbicici", "Code", "seq_classification")

    # Relative Paths
    LARGE_TRAINING_SET = os.path.join(ROOT_PATH, "data", "training-data-large.txt")
    LARGE_TEST_SET = os.path.join(ROOT_PATH, "data", "test-data-large.txt")
    EMBEDDING_CHECKPOINT_PATH = os.path.join(ROOT_PATH, "embedding_training", "embeddings")
    SMALL_TRAINING_SET = os.path.join(ROOT_PATH,  "data", "training-data-small.txt")
    SMALL_TEST_SET = os.path.join(ROOT_PATH, "data", "test-data-small.txt")
    WORD_EMBEDDING_FILE_PATH = os.path.join(ROOT_PATH, "embedding_training", "embeddings", "embedding_epoch24.ckpt")
    RNN_MODEL_CHECKPOINT_PATH = os.path.join(ROOT_PATH, "classifiers", "models")
    CNN_MODEL_CHECKPOINT_PATH = os.path.join(ROOT_PATH, "classifiers", "cnn_models")