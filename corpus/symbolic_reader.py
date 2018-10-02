from auxilliary.multitasking import ThreadWorker
from corpus.sequence import Sequence


class SymbolicReader(ThreadWorker):
    def __init__(self, thread_id, task_list, **kwargs):
        super().__init__(thread_id, task_list)
        self.isTraining = kwargs["is_training"]

    def run(self):
        # Each line is a document
        document_lines = self.taskList
        self.resultList = []
        if self.isTraining:
            for line in document_lines:
                # Split label-sequence
                label_sequence_pair = line.split("\t")
                label = int(label_sequence_pair[0])
                sequence_tokens = label_sequence_pair[1].replace("\n", "").split(",")
                self.resultList.append((label, sequence_tokens))
        else:
            for line in document_lines:
                # Only sequence
                sequence_tokens = line.replace("\n", "").split(",")
                self.resultList.append(sequence_tokens)
