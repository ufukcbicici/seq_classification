import numpy as np


class Sentence:
    def __init__(self, sentence_raw, tokens=None):
        self.rawSentence = sentence_raw
        if tokens is None:
            tokens = []
        self.tokens = tokens


class Document:
    def __init__(self, sentences=None):
        self.sentences = []
        self.label = -1
        self.documentId = None
        if sentences is None:
            sentences = []
        self.tokenSet = set()
        for sentence in sentences:
            self.add_sentence(sentence=sentence)

    def add_sentence(self, sentence):
        self.sentences.append(sentence)
        self.tokenSet = self.tokenSet.union(set(sentence.tokens))

    def set_label(self, label):
        self.label = label


class Sequence:
    def __init__(self, document_id, label, tokens_list, is_training=1):
        self.documentId = document_id
        self.isTraining = is_training
        self.label = label
        self.tokenArr = np.array(tokens_list, dtype=np.int32)


class Corpus:
    def __init__(self):
        self.documents = []
        self.fullVocabularyTermFrequencies = {}
        self.fullVocabularyDocumentFrequencies = {}
        self.fullVocabularyTfIdf = {}
        self.validVocabularyFrequencies = {}
        self.validVocabularyTokenToIndex = {}
        self.validVocabularyIndexToToken = {}
        self.setOfUnknownTokens = set()
        self.embeddingContextsAndTargets = None
        self.currentIndex = 0
        self.currentIndices = None
        self.isNewEpoch = False
        self.currentEpoch = 0
        # Classifier Training
        self.trainingSequences = []
        self.validationSequences = []
        self.currentSequenceList = None
        self.currentIndexList = []
        self.currentSequenceIndex = None

    def read_documents(self, path):
        pass

        # with open(json_path, encoding="utf8") as f:
        #     lines = f.readlines()
        # self.documents = MultiTaskRunner.run_task(runner_type=IdeaCorpusTokenizer, tasks=lines,
        #                                           labeling_func=labeling_func)
        # # Assign document ids
        # document_id = 0
        # for document in self.documents:
        #     document.documentId = document_id
        #     document_id += 1