class Sequence:
    def __init__(self, document_id, label, tokens_list, is_training=1):
        self.documentId = document_id
        self.isTraining = is_training
        self.label = label
        # self.tokenArr = np.array(tokens_list, dtype=np.int32)
        self.tokenArr = tokens_list

    def get_tokens(self, n_grams):
        seq_length = len(self.tokenArr)
        ngram_list = []
        for window in n_grams:
            for i in range(seq_length):
                if i + window - 1 >= seq_length:
                    break
                token = tuple(self.tokenArr[i:i + window])
                ngram_list.append(token)
        return ngram_list
