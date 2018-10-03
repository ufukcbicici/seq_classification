import os


class GlobalConstants:
    THREAD_COUNT = 1
    LARGE_TRAINING_SET = os.path.join("D:\\", "seq_classification", "data", "training-data-large.txt")
    LARGE_TEST_SET = os.path.join("D:\\", "seq_classification", "data", "test-data-large.txt")
    CORPUS_FREQUENCY_THRESHOLD = 3

