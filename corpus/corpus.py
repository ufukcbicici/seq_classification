import numpy as np
from auxilliary.db_logger import DbLogger
from global_constants import GlobalConstants, DatasetType


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
        # Classifier Training
        self.currentSequenceList = None
        self.currentIndexList = []
        self.currentSequenceIndex = None
        self.isNewEpoch = False

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

    def set_current_dataset_type(self, dataset_type):
        if dataset_type == DatasetType.Training:
            self.currentSequenceList = self.trainingSequences
            self.currentIndexList = np.arange(len(self.trainingSequences))
        elif dataset_type == DatasetType.Validation:
            self.currentSequenceList = self.validationSequences
            self.currentIndexList = np.arange(len(self.validationSequences))
        elif dataset_type == DatasetType.Test:
            self.currentSequenceList = self.testSequences
            self.currentIndexList = np.arange(len(self.testSequences))
        np.random.shuffle(self.currentIndexList)
        self.currentSequenceIndex = 0

    def get_next_training_batch(self, batch_size, wrap_around=True):
        num_of_samples = len(self.currentSequenceList)
        curr_end_index = self.currentSequenceIndex + batch_size - 1
        # Check if the interval [curr_start_index, curr_end_index] is inside data boundaries.
        if 0 <= self.currentSequenceIndex and curr_end_index < num_of_samples:
            indices_list = self.currentIndexList[self.currentSequenceIndex:curr_end_index + 1]
        elif self.currentSequenceIndex < num_of_samples <= curr_end_index:
            indices_list = self.currentIndexList[self.currentSequenceIndex:num_of_samples]
            if wrap_around:
                curr_end_index = curr_end_index % num_of_samples
                indices_list = np.concatenate([indices_list, self.currentIndexList[0:curr_end_index + 1]])
        else:
            raise Exception("Invalid index positions: self.currentIndex={0} - curr_end_index={1}"
                            .format(self.currentSequenceIndex, curr_end_index))
        self.currentSequenceIndex = self.currentSequenceIndex + batch_size
        # Prepare minibatch
        batch_lengths = [len(self.currentSequenceList[i].tokenArr) for i in indices_list]
        max_length = max(batch_lengths)
        data = np.zeros(shape=(batch_size, max_length), dtype=np.int32)
        labels = np.zeros(shape=(batch_size,), dtype=np.int32)
        lengths = np.zeros(shape=(batch_size,), dtype=np.int32)
        document_ids = np.zeros(shape=(batch_size,), dtype=np.int32)
        count = 0
        for index in indices_list:
            sequence = self.currentSequenceList[index]
            seq_length = len(sequence.tokenArr)
            lengths[count] = seq_length
            labels[count] = sequence.label
            document_ids[count] = sequence.documentId
            token_indices = np.array(sequence.tokenArr)
            # Increase indices by one to handle zero entries
            token_indices += 1
            data[count, 0:seq_length] = token_indices
            count += 1
        if num_of_samples <= self.currentSequenceIndex:
            self.isNewEpoch = True
            np.random.shuffle(self.currentIndexList)
            if wrap_around:
                self.currentSequenceIndex = self.currentSequenceIndex % num_of_samples
            else:
                self.currentSequenceIndex = 0
        else:
            self.isNewEpoch = False
        return data, labels, lengths, document_ids
