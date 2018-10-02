from auxilliary.db_logger import DbLogger
from auxilliary.multitasking import MultiTaskRunner
from corpus.corpus import Corpus
from corpus.sequence import Sequence
from corpus.symbolic_reader import SymbolicReader
from random import seed
from random import shuffle
import numpy as np


class SymbolicCorpus(Corpus):
    def __init__(self):
        super().__init__()
        seed(42)

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

        # validation_ids = set(
        #     np.random.choice(len(self.trainingSequences), int(len(self.trainingSequences) * validation_set),
        #                      replace=False).tolist())
        # for id in validation_ids:
        #     if id ==
        # sampled_document_sequences = [seq for index, seq in enumerate(document_sequences)
        #                               if index in sample_sequence_ids]
        # all_sequences.extend(sampled_document_sequences)

    def write_vocabularies_to_db(self, training_table, test_table):
        rows = [(k, v) for k, v in self.fullTrainingCorpusFrequencies.items()]
        DbLogger.write_into_table(rows=rows, table=training_table, col_count=2)
        rows = [(k, v) for k, v in self.fullTestCorpusFrequencies.items()]
        DbLogger.write_into_table(rows=rows, table=test_table, col_count=2)
        print("X")

    def analyze_data(self):
        sequences = [self.trainingSequences, self.validationSequences, self.testSequences]
        vocabularies = [self.fullTrainingCorpusFrequencies, self.fullValidationCorpusFrequencies,
                        self.fullTestCorpusFrequencies]
        for sequence_list, vocabulary in zip(sequences, vocabularies):
            for sequence in sequence_list:
                for token in sequence.tokenArr:
                    if token not in vocabulary:
                        vocabulary[token] = 0
                    vocabulary[token] += 1
        print("X")

