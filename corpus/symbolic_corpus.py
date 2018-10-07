from auxilliary.db_logger import DbLogger
from auxilliary.multitasking import MultiTaskRunner
from auxilliary.utility_funcs import UtilityFuncs
from corpus.corpus import Corpus
from corpus.sequence import Sequence
from corpus.symbolic_reader import SymbolicReader
from random import seed
from random import shuffle
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from collections import deque, Counter
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
            for id, sequence in enumerate(sequence_list):
                if id % 50000 == 0:
                    print("{0} sequences have been processed.".format(id))
                for token in sequence.tokenArr:
                    UtilityFuncs.increase_dict_entry(dictionary=vocabulary, key=token, val=1)
        self.find_low_freq_token_clusters()
        for token, freq in self.fullTrainingCorpusFrequencies.items():
            if freq >= GlobalConstants.CORPUS_FREQUENCY_THRESHOLD:
                self.trainingVocabulary[token] = freq
            else:
                new_token = self.assign_token_to_cluster(token=token)
                UtilityFuncs.increase_dict_entry(dictionary=self.trainingVocabulary, key=new_token, val=freq)
        # Add unknown as well
        UtilityFuncs.increase_dict_entry(dictionary=self.trainingVocabulary, key="UNK", val=0)
        # Order according to frequencies
        token_counter = Counter(self.trainingVocabulary)
        ordered_tokens = token_counter.most_common()
        # Delete vocabulary table
        DbLogger.delete_table(table=DbLogger.finalizedVocabularyTable)
        db_rows = []
        for index, tpl in enumerate(ordered_tokens):
            token = tpl[0]
            freq = tpl[1]
            self.vocabularyTokenToIndex[token] = index
            self.vocabularyIndexToToken[index] = token
            db_rows.append((token, index, freq))
        # Write to db
        DbLogger.write_into_table(rows=db_rows, table=DbLogger.finalizedVocabularyTable, col_count=3)
        print("X")

    def assign_token_to_cluster(self, token):
        first_letter = token[0]
        if first_letter not in self.lowFreqTokenClusterCenters:
            return "UNK"
        relevant_cluster_centers = self.lowFreqTokenClusterCenters[first_letter]
        closest_cluster_center = UtilityFuncs.take_closest(sorted_list=relevant_cluster_centers, val=int(token[1:]))
        new_token = "{0}_lowfreq_{1}".format(first_letter, closest_cluster_center)
        return new_token

    def get_token_id(self, token):
        if token in self.vocabularyTokenToIndex:
            return self.vocabularyTokenToIndex[token]
        else:
            new_token = self.assign_token_to_cluster(token=token)
            return self.vocabularyTokenToIndex[new_token]

    # Private methods
    def find_low_freq_token_clusters(self):
        first_letters = set([token[0] for token in self.fullTrainingCorpusFrequencies.keys()])
        low_freq_tokens = [token for token, freq in self.fullTrainingCorpusFrequencies.items()
                           if freq < GlobalConstants.CORPUS_FREQUENCY_THRESHOLD]
        numeric_codes_dict = {}
        for letter in first_letters:
            print("Processing letter:{0}".format(letter))
            numeric_codes_dict[letter] = []
            for token in low_freq_tokens:
                if token[0] == letter:
                    numeric_codes_dict[letter].append(int(token[1:]))
            numeric_codes = numeric_codes_dict[letter]
            if len(numeric_codes) == 1:
                self.lowFreqTokenClusterCenters[letter] = [numeric_codes[0]]
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
                    ms.fit(sub_cluster)
                    labels = np.unique(ms.labels_)
                    cluster_centers = ms.cluster_centers_
                    if len(labels) == 1:
                        cluster_center = cluster_centers[0]
                        self.lowFreqTokenClusterCenters[letter].append(np.asscalar(cluster_center))
                    else:
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
                                self.lowFreqTokenClusterCenters[letter].append(np.asscalar(cluster_center))
                                print("New cluster center:{0}".format(cluster_center))
                # Sort all cluster centers
                self.lowFreqTokenClusterCenters[letter] = sorted(self.lowFreqTokenClusterCenters[letter])
