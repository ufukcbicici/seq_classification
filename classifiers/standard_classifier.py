import tensorflow as tf
import numpy as np
import os
from collections import Counter
from auxilliary.db_logger import DbLogger
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from global_constants import GlobalConstants, DatasetType \
    # , DocumentLabel


class StandardClassifer:
    def __init__(self, corpus):
        tf.reset_default_graph()
        self.corpus = corpus

    def train(self):
        pass

    def test(self, dataset_type):
        pass