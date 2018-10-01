from corpus.corpus import Corpus
from sklearn.feature_extraction.text import CountVectorizer


class SymbolicCorpus(Corpus):
    def __init__(self):
        super().__init__()

    def read_documents(self, path):
        with open(path, encoding="utf8") as f:
            lines = f.readlines()
        self.documents = MultiTaskRunner.run_task(runner_type=IdeaCorpusTokenizer, tasks=lines,
                                                  labeling_func=labeling_func)