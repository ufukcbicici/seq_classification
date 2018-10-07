import tensorflow as tf
import numpy as np
import os
from collections import Counter
from auxilliary.db_logger import DbLogger
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from global_constants import GlobalConstants, DatasetType \
    # , DocumentLabel


class DeepClassifier:
    def __init__(self, corpus):
        tf.reset_default_graph()
        self.corpus = corpus
        # Load embeddings
        pretrained_var_list = checkpoint_utils.list_variables(GlobalConstants.WORD_EMBEDDING_FILE_PATH)
        self.wordEmbeddings = checkpoint_utils.load_variable(checkpoint_dir=GlobalConstants.WORD_EMBEDDING_FILE_PATH,
                                                             name="embeddings")
        # IMPORTANT !!! Add a zero row at the 0. position. This will be used as the padding feature.
        self.wordEmbeddings = np.concatenate([np.zeros(shape=(1, GlobalConstants.EMBEDDING_SIZE)), self.wordEmbeddings],
                                             axis=0)
        assert self.wordEmbeddings.shape[0] == self.corpus.get_vocabulary_size() + 1
        assert self.wordEmbeddings.shape[1] == GlobalConstants.EMBEDDING_SIZE
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.input_word_codes = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_word_codes')
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None], name='input_y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')
        self.max_sequence_length = tf.placeholder(dtype=tf.int32, name='max_sequence_length')
        self.isTrainingFlag = tf.placeholder(name="is_training", dtype=tf.bool)
        self.embeddings = None
        self.inputs = None
        self.logits = None
        self.predictions = None
        self.numOfCorrectPredictions = None
        self.accuracy = None
        self.optimizer = None
        self.globalStep = None
        self.sess = None
        self.correctPredictions = None
        # L2 loss
        self.mainLoss = None
        self.l2_loss = tf.constant(0.0)

    def build_classifier(self):
        self.get_embeddings()
        self.get_classifier_structure()
        self.get_softmax_layer()
        self.get_loss()
        self.get_accuracy()
        self.get_optimizer()
        self.sess = tf.Session()

    def get_embeddings(self):
        # vocabulary_size = self.corpus.get_vocabulary_size()
        # embedding_size = GlobalConstants.EMBEDDING_SIZE
        # max_sequence_length = GlobalConstants.MAX_SEQUENCE_LENGTH
        with tf.name_scope('embedding'):
            self.embeddings = tf.get_variable('embedding',
                                              shape=[self.wordEmbeddings.shape[0], self.wordEmbeddings.shape[1]],
                                              dtype=tf.float32, trainable=GlobalConstants.FINE_TUNE_EMBEDDINGS)
            self.inputs = tf.nn.embedding_lookup(self.embeddings, self.input_word_codes)

    def get_classifier_structure(self):
        pass

    def get_softmax_layer(self):
        pass

    def get_loss(self):
        # Loss
        with tf.name_scope('loss'):
            tvars = tf.trainable_variables()
            # L2 regularization for LSTM weights
            for tv in tvars:
                if 'kernel' in tv.name:
                    self.l2_loss += tf.nn.l2_loss(tv)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                    logits=self.logits)
            self.mainLoss = tf.reduce_mean(losses) + GlobalConstants.L2_LAMBDA_COEFFICENT * self.l2_loss

    def get_accuracy(self):
        with tf.name_scope('accuracy'):
            self.correctPredictions = tf.equal(self.predictions, self.input_y)
            self.numOfCorrectPredictions = tf.reduce_sum(tf.cast(self.correctPredictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(self.correctPredictions, tf.float32), name='accuracy')

    def get_optimizer(self):
        # Train procedure
        self.globalStep = tf.Variable(0, name='global_step', trainable=False)
        # Learning rate decay
        starter_learning_rate = GlobalConstants.INITIAL_LR_CLASSIFIER
        # learning_rate = tf.train.exponential_decay(starter_learning_rate,
        #                                            self.globalStep,
        #                                            GlobalConstants.DECAY_PERIOD_LSTM,
        #                                            GlobalConstants.DECAY_RATE_LSTM,
        #                                            staircase=True)
        self.optimizer = tf.train.AdamOptimizer(starter_learning_rate).minimize(self.mainLoss,
                                                                                global_step=self.globalStep)

    def train(self):
        pass

    def test(self, dataset_type):
        confusion_dict = {}
        batch_size = GlobalConstants.BATCH_SIZE
        if dataset_type == DatasetType.Validation:
            self.corpus.set_current_dataset_type(dataset_type=DatasetType.Validation)
        else:
            self.corpus.set_current_dataset_type(dataset_type=DatasetType.Training)
        total_correct_count = 0
        total_count = 0
        total_correct_document_count = 0
        total_document_count = 0
        document_predictions = {}
        document_correct_labels = {}
        while True:
            data, labels, lengths, document_ids, max_length = \
                self.corpus.get_next_training_batch(batch_size=batch_size, wrap_around=False)
            feed_dict = {self.batch_size: batch_size,
                         self.input_word_codes: data,
                         self.input_y: labels,
                         self.keep_prob: 1.0,
                         self.sequence_length: lengths,
                         self.isTrainingFlag: False}
            run_ops = [self.correctPredictions, self.predictions]
            results = self.sess.run(run_ops, feed_dict=feed_dict)
            for sample_id, document_id in enumerate(document_ids.tolist()):
                if lengths[sample_id] == 0:
                    break
                is_sample_correct = results[0][sample_id]
                sample_prediction = results[1][sample_id]
                total_correct_count += is_sample_correct
                total_count += 1
                if document_id not in document_predictions:
                    document_predictions[document_id] = []
                if document_id not in document_correct_labels:
                    document_correct_labels[document_id] = []
                correct_label = labels[sample_id]
                document_predictions[document_id].append(sample_prediction)
                document_correct_labels[document_id].append(correct_label)
            if self.corpus.isNewEpoch:
                break
        accuracy = float(total_correct_count) / float(total_count)
        print("Dataset:{0} Accuracy:{1}".format(dataset_type, accuracy))
        # Check document correctness
        for k, v in document_correct_labels.items():
            label_set = set(v)
            assert len(label_set) == 1
            document_label = list(label_set)[0]
            prediction_list = document_predictions[k]
            tpl = Counter(prediction_list).most_common(1)
            predicted_label = tpl[0][0]
            if predicted_label == document_label:
                total_correct_document_count += 1
            if (document_label, predicted_label) not in confusion_dict:
                confusion_dict[(document_label, predicted_label)] = 0
            confusion_dict[(document_label, predicted_label)] += 1
            total_document_count += 1
        document_wise_accuracy = float(total_correct_document_count) / float(total_document_count)
        print("Dataset:{0} Document-Wise Accuracy:{1}".format(dataset_type, document_wise_accuracy))
        print("Confusion Matrix:")
        print(confusion_dict)
        return accuracy, document_wise_accuracy

    # def test_compare_with_set(self, dataset_type):
    #     confusion_dict = {}
    #     label_sets = {0: {DocumentLabel.Company, DocumentLabel.Finance}, 1: {DocumentLabel.Other}}
    #     # label_mappings = {DocumentLabel.Company: 0, DocumentLabel.Finance: 0, DocumentLabel.Other: 1}
    #     label_mappings = {DocumentLabel.Company: 0, DocumentLabel.Finance: 1, DocumentLabel.Other: 2}
    #     batch_size = GlobalConstants.BATCH_SIZE
    #     if dataset_type == DatasetType.Validation:
    #         self.corpus.set_current_dataset_type(dataset_type=DatasetType.Validation)
    #     else:
    #         self.corpus.set_current_dataset_type(dataset_type=DatasetType.Training)
    #     total_correct_count = 0
    #     total_count = 0
    #     total_correct_document_count = 0
    #     total_document_count = 0
    #     document_predictions = {}
    #     document_correct_labels = {}
    #     while True:
    #         data, labels, lengths, document_ids = \
    #             self.corpus.get_next_training_batch(batch_size=batch_size, wrap_around=False)
    #         feed_dict = {self.batch_size: batch_size,
    #                      self.input_word_codes: data,
    #                      self.input_y: labels,
    #                      self.keep_prob: 1.0,
    #                      self.sequence_length: lengths,
    #                      self.isTrainingFlag: False}
    #         run_ops = [self.correctPredictions, self.predictions]
    #         results = self.sess.run(run_ops, feed_dict=feed_dict)
    #         for sample_id, document_id in enumerate(document_ids.tolist()):
    #             if lengths[sample_id] == 0:
    #                 break
    #             true_label = labels[sample_id] # label_mappings[labels[sample_id]]
    #             is_sample_correct = results[0][sample_id]
    #             sample_prediction = results[1][sample_id] # label_mappings[results[1][sample_id]]
    #             total_correct_count += is_sample_correct # int(true_label == sample_prediction)
    #             total_count += 1
    #             if document_id not in document_predictions:
    #                 document_predictions[document_id] = []
    #             if document_id not in document_correct_labels:
    #                 document_correct_labels[document_id] = []
    #             document_predictions[document_id].append(sample_prediction)
    #             document_correct_labels[document_id].append(true_label)
    #         if self.corpus.isNewEpoch:
    #             break
    #     accuracy = float(total_correct_count) / float(total_count)
    #     print("Dataset:{0} Accuracy:{1}".format(dataset_type, accuracy))
    #     # Check document correctness
    #     for k, v in document_correct_labels.items():
    #         label_set = set(v)
    #         assert len(label_set) == 1
    #         document_label = list(label_set)[0]
    #         prediction_list = document_predictions[k]
    #         tpl = Counter(prediction_list).most_common(1)
    #         predicted_label = tpl[0][0]
    #         if predicted_label == document_label:
    #             total_correct_document_count += 1
    #         if (document_label, predicted_label) not in confusion_dict:
    #             confusion_dict[(document_label, predicted_label)] = 0
    #         confusion_dict[(document_label, predicted_label)] += 1
    #         total_document_count += 1
    #     document_wise_accuracy = float(total_correct_document_count) / float(total_document_count)
    #     print("Dataset:{0} Document-Wise Accuracy:{1}".format(dataset_type, document_wise_accuracy))
    #     print("Confusion Matrix:")
    #     print(confusion_dict)
    #     return accuracy, document_wise_accuracy

    @staticmethod
    def get_explanation():
        vars_dict = vars(GlobalConstants)
        explanation = ""
        for k, v in vars_dict.items():
            explanation += "{0}: {1}\n".format(k, v)
        return explanation
