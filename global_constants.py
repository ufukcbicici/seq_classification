import os


class GlobalConstants:
    THREAD_COUNT = 1
    # Idea GPU
    # LARGE_TRAINING_SET = os.path.join("D:\\", "deep", "seq_classification", "data", "training-data-large.txt")
    # LARGE_TEST_SET = os.path.join("D:\\", "deep", "seq_classification", "data", "test-data-large.txt")
    # Home
    LARGE_TRAINING_SET = os.path.join("D:\\", "seq_classification", "data", "training-data-large.txt")
    LARGE_TEST_SET = os.path.join("D:\\", "seq_classification", "data", "test-data-large.txt")
    CORPUS_FREQUENCY_THRESHOLD = 3
    MAX_CLUSTER_FREQ_RATIO = 0.25

