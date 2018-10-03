from auxilliary.db_logger import DbLogger
from auxilliary.multitasking import MultiTaskRunner
from corpus.corpus import Corpus
from corpus.sequence import Sequence
from corpus.symbolic_reader import SymbolicReader
from random import seed
from random import shuffle
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from collections import deque
from global_constants import GlobalConstants


class SymbolicCorpus(Corpus):
    def __init__(self):
        super().__init__()
        seed(42)
        self.lowFreqTokenClusterCenters = {}

    def read_documents(self, path, is_training):
        sequence_list = self.trainingSequences if is_training else self.testSequences
        with open(path, encoding="utf8") as f:
            lines = f.readlines()
        results = MultiTaskRunner.run_task(runner_type=SymbolicReader, tasks=lines, is_training=is_training)
        for id, res in enumerate(results):
            sequence = Sequence(document_id=id, label=res[0], tokens_list=res[1], is_training=is_training) \
                if is_training else Sequence(document_id=id, label=-1, tokens_list=res, is_training=is_training)
            sequence_list.append(sequence)
            # # Self process
            # sequence_list = self.trainingSequences if is_training else self.testSequences
            # corpus_token_dict = self.fullTrainingCorpusFrequencies if is_training else self.fullTestCorpusFrequencies
            # results = MultiTaskRunner.run_task(runner_type=SymbolicReader, tasks=lines, is_training=is_training)
            # for id, res in enumerate(results):
            #     sequence = Sequence(document_id=id, label=res[0], tokens_list=res[1], is_training=is_training) \
            #         if is_training else Sequence(document_id=id, label=-1, tokens_list=res, is_training=is_training)
            #     sequence_list.append(sequence)
            #     for token in sequence.tokenArr:
            #         if token not in corpus_token_dict:
            #             corpus_token_dict[token] = 0
            #         corpus_token_dict[token] += 1
            # print("X")

    def pick_validation_set(self, validation_ratio):
        shuffle(self.trainingSequences)
        max_id = int(len(self.trainingSequences) * validation_ratio)
        self.validationSequences = self.trainingSequences[0:max_id]
        del self.trainingSequences[0:max_id]
        print("X")

    def write_vocabularies_to_db(self, training_table, test_table):
        rows = [(k, v) for k, v in self.fullTrainingCorpusFrequencies.items()]
        DbLogger.write_into_table(rows=rows, table=training_table, col_count=2)
        rows = [(k, v) for k, v in self.fullTestCorpusFrequencies.items()]
        DbLogger.write_into_table(rows=rows, table=test_table, col_count=2)
        print("X")

    def analyze_data(self):
        # Build Vocabularies
        sequences = [self.trainingSequences, self.validationSequences, self.testSequences]
        vocabularies = [self.fullTrainingCorpusFrequencies, self.fullValidationCorpusFrequencies,
                        self.fullTestCorpusFrequencies]
        for sequence_list, vocabulary in zip(sequences, vocabularies):
            for sequence in sequence_list:
                for token in sequence.tokenArr:
                    if token not in vocabulary:
                        vocabulary[token] = 0
                    vocabulary[token] += 1
        # Build vocabulary
        first_letters = set([token[0] for token in self.fullTrainingCorpusFrequencies.keys()])
        low_freq_tokens = [token for token, freq in self.fullTrainingCorpusFrequencies.items()
                           if freq < GlobalConstants.CORPUS_FREQUENCY_THRESHOLD]
        numeric_codes_dict = {}
        for letter in first_letters:
            numeric_codes_dict[letter] = []
            for token in low_freq_tokens:
                if token[0] == letter:
                    numeric_codes_dict[letter].append(int(token[1:]))
            numeric_codes = numeric_codes_dict[letter]
            if len(numeric_codes) == 1:
                self.lowFreqTokenClusterCenters[letter] = np.array(numeric_codes[0])
                print("X")
            else:
                self.lowFreqTokenClusterCenters[letter] = []
                # Divide into partitions recursively until all clusters have a freq < TOTAL_COUNT*MAX_CLUSTER_FREQ_RATIO
                numeric_arr = np.array(numeric_codes).reshape(len(numeric_codes), 1)
                freq_threshold = int(float(numeric_arr.shape[0]) * GlobalConstants.MAX_CLUSTER_FREQ_RATIO)
                cluster_info_tpls = deque()
                cluster_info_tpls.append(numeric_arr)
                while len(cluster_info_tpls) > 0:
                    sub_cluster = cluster_info_tpls.popleft()
                    bandwidth = estimate_bandwidth(sub_cluster)
                    ms = MeanShift(bandwidth=bandwidth)
                    ms.fit(numeric_arr)
                    labels = np.unique(ms.labels_)
                    cluster_centers = ms.cluster_centers_
                    for label in labels:
                        label_mask = ms.labels_ == label
                        new_cluster_freq = np.sum(label_mask)
                        # Cluster is too big.
                        if new_cluster_freq >= freq_threshold:
                            # Select members with the given label
                            new_cluster = sub_cluster[np.nonzero(label_mask)]
                            cluster_info_tpls.append(new_cluster)
                        # Cluster is small enough
                        else:
                            cluster_center = cluster_centers[label]
                            self.lowFreqTokenClusterCenters[letter].append(cluster_center)
                        print("X")
                # numeric_arr = np.array(numeric_codes).reshape(len(numeric_codes), 1)
                # bandwidth = estimate_bandwidth(numeric_arr)
                # ms = MeanShift(bandwidth=bandwidth)
                # ms.fit(numeric_arr)
                # labels = ms.labels_
                # cluster_centers = ms.cluster_centers_
                # self.lowFreqTokenClusterCenters[letter] = cluster_centers.reshape(cluster_centers.shape[0])
                # print("X")
                # if len((numeric_codes_dict[letter])        :
        # Apply mean-shift
        print("X")

