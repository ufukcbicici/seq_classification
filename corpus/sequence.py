class Sequence:
    def __init__(self, document_id, label, tokens_list, is_training=1):
        self.documentId = document_id
        self.isTraining = is_training
        self.label = label
        # self.tokenArr = np.array(tokens_list, dtype=np.int32)
        self.tokenArr = tokens_list
