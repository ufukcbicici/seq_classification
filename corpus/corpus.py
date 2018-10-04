import numpy as np
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

    def read_vocabulary(self):
        rows = DbLogger.read_tuples_from_table(table_name=DbLogger.finalizedVocabularyTable)
        for row in rows:
            token = row[0]
            index = row[1]
            tf = row[2]
            self.vocabularyTokenToIndex[token] = index
            self.vocabularyIndexToToken[index] = token
        print("X")

    def get_vocabulary_size(self):
        return len(self.vocabularyTokenToIndex)
