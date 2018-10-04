import numpy as np


# class Sentence:
#     def __init__(self, sentence_raw, tokens=None):
#         self.rawSentence = sentence_raw
#         if tokens is None:
#             tokens = []
#         self.tokens = tokens
#
#
# class Document:
#     def __init__(self, sentences=None):
#         self.sentences = []
#         self.label = -1
#         self.documentId = None
#         if sentences is None:
#             sentences = []
#         self.tokenSet = set()
#         for sentence in sentences:
#             self.add_sentence(sentence=sentence)
#
#     def add_sentence(self, sentence):
#         self.sentences.append(sentence)
#         self.tokenSet = self.tokenSet.union(set(sentence.tokens))
#
#     def set_label(self, label):
#         self.label = label
from auxilliary.db_logger import DbLogger
from global_constants import GlobalConstants


class Corpus:
    def __init__(self):
        self.trainingSequences = []
        self.validationSequences = []
        self.testSequences = []
        self.fullTrainingCorpusFrequencies = {}
        self.fullValidationCorpusFrequencies = {}
        self.fullTestCorpusFrequencies = {}
        self.trainingVocabulary = {}
        self.vocabularyTokenToIndex = {}
        self.vocabularyIndexToToken = {}

    def read_documents(self, path, is_training):
        pass

    def pick_validation_set(self, validation_set):
        pass

    def write_vocabularies_to_db(self, training_table, test_table):
        pass

    def analyze_data(self):
        pass

    def get_token_id(self, token):
        pass

    def build_cbow_tokens(self):
        rows = []
        table_name = "cbow_skip_window_{0}_table".format(GlobalConstants.CBOW_WINDOW_SIZE)
        # Delete cbow table
        DbLogger.delete_table(table=table_name)
        sequence_count = 0
        for sequence in self.trainingSequences:
            for token_index in range(len(sequence.tokenArr)):
                context = []
                target_token = sequence.tokenArr[token_index]
                target_id = self.get_token_id(token=target_token)
                # Harvest a context
                for delta in range(-GlobalConstants.CBOW_WINDOW_SIZE, GlobalConstants.CBOW_WINDOW_SIZE + 1):
                    t = delta + token_index
                    if t < 0 or t >= len(sequence.tokenArr):
                        context.append(-1)
                    elif t == token_index:
                        assert target_token == sequence.tokenArr[token_index]
                        continue
                    else:
                        token = sequence.tokenArr[t]
                        tokend_id = self.get_token_id(token=token)
                        context.append(tokend_id)
                context.append(target_id)
                assert len(context) == 2 * GlobalConstants.CBOW_WINDOW_SIZE + 1
                rows.append(tuple(context))
            sequence_count += 1
            if sequence_count % 10000 == 0:
                print("{0} sequences have been processed.".format(sequence_count))
            if len(rows) >= 100000:
                print("CBOW tokens written to DB.")
                DbLogger.write_into_table(rows=rows, table=table_name,
                                          col_count=2 * GlobalConstants.CBOW_WINDOW_SIZE + 1)
                rows = []
        if len(rows) > 0:
            DbLogger.write_into_table(rows=rows, table=table_name,
                                      col_count=2 * GlobalConstants.CBOW_WINDOW_SIZE + 1)
