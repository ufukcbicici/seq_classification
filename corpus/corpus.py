import numpy as np
from auxilliary.db_logger import DbLogger
from auxilliary.utility_funcs import UtilityFuncs
from global_constants import GlobalConstants, DatasetType
from corpus.sequence import Sequence
from collections import deque, Counter


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
        self.vocabularyTokenToFreq = {}
        # Classifier Training
        self.currentSequenceList = None
        self.currentIndexList = []
        self.currentSequenceIndex = None
        self.isNewEpoch = False
        self.tfIdfFeatures = {}
        self.labelsDict = {}

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
            token_id_list = [self.get_token_id(token=token) for token in sequence.tokenArr]
            token_indices = np.array(token_id_list)
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
        return data, labels, lengths, document_ids, max_length

    def get_num_of_classes(self):
        labels = set([seq.label for seq in self.trainingSequences])
        return len(labels)

    def split_dataset_into_sequences(self):
        datasets = [DatasetType.Training, DatasetType.Validation, DatasetType.Test]
        training_flags = [1, 0, 0]
        for dataset_index, dataset_type in enumerate(datasets):
            splitted_dataset = []
            training_flag = training_flags[dataset_index]
            if dataset_type == DatasetType.Training:
                dataset = self.trainingSequences
            elif dataset_type == DatasetType.Validation:
                dataset = self.validationSequences
            else:
                dataset = self.testSequences
            for sequence in dataset:
                curr_index = 0
                while True:
                    subsequence = sequence.tokenArr[curr_index:curr_index + GlobalConstants.MAX_SEQUENCE_LENGTH]
                    if len(subsequence) == 0:
                        break
                    splitted_dataset.append(Sequence(document_id=sequence.documentId, label=sequence.label,
                                                     tokens_list=subsequence, is_training=training_flag))
                    curr_index += GlobalConstants.SEQUENCE_SLIDING_WINDOW_SIZE
            if dataset_type == DatasetType.Training:
                self.trainingSequences = splitted_dataset
            elif dataset_type == DatasetType.Validation:
                self.validationSequences = splitted_dataset
            else:
                self.testSequences = splitted_dataset
        print("X")

    def tf_idf_analysis(self):
        # Step 1: Extract the vocabulary
        corpus_freq_dict = {}
        idf_dict = {}
        bag_of_words = set()
        for sequence in self.trainingSequences:
            document_token_set = set(sequence.get_tokens(n_grams=GlobalConstants.N_GRAMS))
            # Build the bag of words
            bag_of_words = bag_of_words.union(document_token_set)
            # Add to corpus frequency dict, will be used for Inverse Term Frequencies (Idf)
            for token in document_token_set:
                UtilityFuncs.increase_dict_entry(dictionary=corpus_freq_dict, key=token, val=1)
        # Calculate idf values
        corpus_size = float(len(self.trainingSequences))
        for k, v in corpus_freq_dict.items():
            idf_dict[k] = np.log10(corpus_size / float(v))
        # Step 2: Build features for training, validation and test sets
        token_dict = {}
        id = 0
        for token in bag_of_words:
            token_dict[token] = id
            id += 1
        datasets = [self.trainingSequences, self.validationSequences, self.testSequences]
        dataset_types = [DatasetType.Training, DatasetType.Validation, DatasetType.Test]
        for dataset, dataset_type in zip(datasets, dataset_types):
            self.tfIdfFeatures[dataset_type] = np.zeros((len(dataset), len(token_dict)), dtype=np.float32)
            if dataset_type == DatasetType.Training or dataset_type == DatasetType.Validation:
                self.labelsDict[dataset_type] = np.zeros(shape=(len(dataset), ), dtype=np.int32)
            for id, sequence in enumerate(dataset):
                tokens = sequence.get_tokens(n_grams=GlobalConstants.N_GRAMS)
                token_counter = Counter(tokens)
                ordered_tokens = token_counter.most_common()
                for tpl in ordered_tokens:
                    # Skip unknown tokens
                    if tpl[0] not in token_dict:
                        continue
                    coordinate = token_dict[tpl[0]]
                    # Get Tf-Idf
                    tf = tpl[1]
                    idf = idf_dict[tpl[0]]
                    self.tfIdfFeatures[dataset_type][id, coordinate] = float(tf)*idf
                if dataset_type == DatasetType.Training or dataset_type == DatasetType.Validation:
                    self.labelsDict[dataset_type][id] = sequence.label
        print("X")


