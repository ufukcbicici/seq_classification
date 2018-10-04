from auxilliary.db_logger import DbLogger
from global_constants import GlobalConstants
import numpy as np


class ContextGenerator:
    def __init__(self):
        self.currentIndex = 0
        self.currentIndices = None
        self.isNewEpoch = False
        self.embeddingContextsAndTargets = None

    @staticmethod
    def build_cbow_tokens(corpus):
        rows = []
        table_name = "cbow_skip_window_{0}_table".format(GlobalConstants.CBOW_WINDOW_SIZE)
        # Delete cbow table
        DbLogger.delete_table(table=table_name)
        sequence_count = 0
        for sequence in corpus.trainingSequences:
            for token_index in range(len(sequence.tokenArr)):
                context = []
                target_token = sequence.tokenArr[token_index]
                target_id = corpus.get_token_id(token=target_token)
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
                        tokend_id = corpus.get_token_id(token=token)
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

    def read_cbow_data(self):
        table_name = "cbow_skip_window_{0}_table".format(GlobalConstants.CBOW_WINDOW_SIZE)
        condition = ""
        for i in range(2 * GlobalConstants.CBOW_WINDOW_SIZE):
            condition += "Token{0} != -1".format(i)
            if i < 2 * GlobalConstants.CBOW_WINDOW_SIZE - 1:
                condition += " AND "
        rows = DbLogger.read_tuples_from_table(table_name=table_name, condition=condition)
        self.embeddingContextsAndTargets = np.zeros(shape=(len(rows), 2 * GlobalConstants.CBOW_WINDOW_SIZE + 1),
                                                    dtype=np.int32)
        for i in range(len(rows)):
            row = rows[i]
            for j in range(2 * GlobalConstants.CBOW_WINDOW_SIZE):
                self.embeddingContextsAndTargets[i, j] = row[j]
            self.embeddingContextsAndTargets[i, -1] = row[-1]
        self.reset()
        print("X")

    def reset(self):
        self.currentIndex = 0
        self.currentIndices = np.arange(self.embeddingContextsAndTargets.shape[0])
        np.random.shuffle(self.currentIndices)
        self.isNewEpoch = False

    def get_next_batch(self, batch_size):
        num_of_samples = len(self.embeddingContextsAndTargets)
        curr_end_index = self.currentIndex + batch_size - 1
        # Check if the interval [curr_start_index, curr_end_index] is inside data boundaries.
        if 0 <= self.currentIndex and curr_end_index < num_of_samples:
            indices_list = self.currentIndices[self.currentIndex:curr_end_index + 1]
        elif self.currentIndex < num_of_samples <= curr_end_index:
            indices_list = self.currentIndices[self.currentIndex:num_of_samples]
            curr_end_index = curr_end_index % num_of_samples
            indices_list = np.concatenate([indices_list, self.currentIndices[0:curr_end_index + 1]])
        else:
            raise Exception("Invalid index positions: self.currentIndex={0} - curr_end_index={1}"
                            .format(self.currentIndex, curr_end_index))
        self.currentIndex = self.currentIndex + batch_size
        context = self.embeddingContextsAndTargets[indices_list, 0:2 * GlobalConstants.CBOW_WINDOW_SIZE]
        targets = self.embeddingContextsAndTargets[indices_list, -1]
        if num_of_samples <= self.currentIndex:
            self.isNewEpoch = True
            np.random.shuffle(self.currentIndices)
            self.currentIndex = self.currentIndex % num_of_samples
        else:
            self.isNewEpoch = False
        return context, targets

    def validate(self, corpus, embeddings):
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        embedding_norms = np.reshape(embedding_norms, newshape=(embedding_norms.shape[0], 1))
        embeddings = embeddings / embedding_norms
        validation_tokens = GlobalConstants.VALIDATION_TOKENS
        for token in validation_tokens:
            token_id = corpus.get_token_id(token=token)
            token_embedding = embeddings[token_id]
            token_embedding = np.reshape(token_embedding, newshape=(token_embedding.shape[0], 1))
            embedding_cosines = np.dot(embeddings, token_embedding)
            embedding_cosines = np.reshape(embedding_cosines, newshape=(embedding_cosines.shape[0],))
            sorted_indices = np.argsort(embedding_cosines)[::-1]
            print_str = "{0}   :".format(token)
            for i in range(10):
                print_str += "{0},".format(corpus.vocabularyIndexToToken[sorted_indices[i]])
            print_str = print_str[0:len(print_str) - 1]
            print(print_str)