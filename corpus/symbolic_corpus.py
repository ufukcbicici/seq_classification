from auxilliary.multitasking import MultiTaskRunner
from corpus.corpus import Corpus
from sklearn.feature_extraction.text import CountVectorizer
from corpus.sequence import Sequence
from corpus.symbolic_reader import SymbolicReader


class SymbolicCorpus(Corpus):
    def __init__(self):
        super().__init__()

    def read_documents(self, path, is_training):
        sequence_list = self.trainingSequences if is_training else self.testSequences
        corpus_token_dict = self.fullTrainingCorpusFrequencies if is_training else self.fullTestCorpusFrequencies
        with open(path, encoding="utf8") as f:
            lines = f.readlines()
        results = MultiTaskRunner.run_task(runner_type=SymbolicReader, tasks=lines, is_training=is_training)
        for id, res in enumerate(results):
            sequence = Sequence(document_id=id, label=res[0], tokens_list=res[1], is_training=is_training) \
                if is_training else Sequence(document_id=id, label=-1, tokens_list=res, is_training=is_training)
            sequence_list.append(sequence)
            for token in sequence.tokenArr:
                if token not in corpus_token_dict:
                    corpus_token_dict[token] = 0
                    corpus_token_dict[token] += 1
        print("X")