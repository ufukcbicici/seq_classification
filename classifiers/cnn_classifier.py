from auxilliary.db_logger import DbLogger

from collections import Counter

from classifiers.deep_classifier import DeepClassifier
from global_constants import GlobalConstants, DatasetType
import tensorflow as tf
import numpy as np
import os


class CnnClassifier(DeepClassifier):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.conv_outputs = None
        self.padded_conv_outputs = None
        self.flattened_convs = None
        self.classifier_output = None
        self.concatenated_output = None
        self.imageForm = None
        self.embeddingImage = None
        self.finalFeature = None

    def get_embeddings(self):
        super().get_embeddings()
        self.inputs = tf.expand_dims(self.inputs, axis=3)

    def get_classifier_structure(self):
        # Embedding Convolution Layer
        embedding_size = GlobalConstants.EMBEDDING_SIZE
        self.conv_outputs = {}
        self.padded_conv_outputs = {}
        total_dim = 0
        for window_length in GlobalConstants.CNN_SLIDING_WINDOWS:
            net_branch = tf.layers.conv2d(
                self.inputs,
                filters=1,
                kernel_size=(window_length, embedding_size),
                padding='valid',
                strides=(1, 1),
                activation=tf.nn.relu,
                use_bias=True)
            total_dim += GlobalConstants.MAX_SEQUENCE_LENGTH - window_length + 1
            self.conv_outputs[window_length] = net_branch
            pad_count = window_length - 1
            if pad_count == 0:
                self.padded_conv_outputs[window_length] = self.conv_outputs[window_length]
            else:
                self.padded_conv_outputs[window_length] = tf.pad(self.conv_outputs[window_length],
                                                                 paddings=tf.constant(
                                                                     [[0, 0], [0, pad_count], [0, 0], [0, 0]]))
        # Build Image
        self.embeddingImage = tf.concat(list(self.padded_conv_outputs.values()), axis=2)
        self.embeddingImage = tf.reshape(self.embeddingImage, shape=[GlobalConstants.BATCH_SIZE,
                                                                     GlobalConstants.MAX_SEQUENCE_LENGTH,
                                                                     len(GlobalConstants.CNN_SLIDING_WINDOWS), 1])
        # Convolutional Layers
        # Layer 1
        net = tf.layers.conv2d(
            self.embeddingImage,
            filters=128,
            kernel_size=3,
            padding='same',
            strides=1,
            activation=None,
            use_bias=False)
        net = tf.layers.batch_normalization(net, training=self.isTrainingFlag)
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Layer 2
        net = tf.layers.conv2d(
            net,
            filters=256,
            kernel_size=3,
            padding='same',
            strides=1,
            activation=None,
            use_bias=False)
        net = tf.layers.batch_normalization(net, training=self.isTrainingFlag)
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Layer 3
        net = tf.layers.conv2d(
            net,
            filters=512,
            kernel_size=3,
            padding='same',
            strides=1,
            activation=None,
            use_bias=False)
        net = tf.layers.batch_normalization(net, training=self.isTrainingFlag)
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Layer 4
        net = tf.layers.conv2d(
            net,
            filters=1024,
            kernel_size=3,
            padding='same',
            strides=1,
            activation=None,
            use_bias=False)
        net = tf.layers.batch_normalization(net, training=self.isTrainingFlag)
        net = tf.nn.relu(net)
        # Global Average Pooling
        net_shape = net.get_shape().as_list()
        net = tf.nn.avg_pool(
            net,
            ksize=[1, net_shape[1], net_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID')
        net = tf.contrib.layers.flatten(net)
        # Dense Layers
        net = tf.layers.dense(net, 2048, activation=tf.nn.relu)
        net = tf.nn.dropout(net, keep_prob=self.keep_prob)
        net = tf.layers.dense(net, 2048, activation=tf.nn.relu)
        self.finalFeature = tf.nn.dropout(net, keep_prob=self.keep_prob)
        # self.finalFeature = ResnetGenerator.build_network(input=self.embeddingImage,
        #                                                   training_flag=self.isTrainingFlag,
        #                                                   groups=ResnetGenerator.RESNET_23,
        #                                                   first_max_pool_size=(2, 2),
        #                                                   first_max_pool_stride=(2, 2),
        #                                                   first_conv_stride=1,
        #                                                   first_conv_filter_count=64,
        #                                                   first_conv_kernel_size=7,
        #                                                   use_batch_norm=True)
        # print("X")
        # FC Output
        # FC Layers
        # self.flattened_convs = {}
        # for window_length, conv_outputs in self.conv_outputs.items():
        #     self.flattened_convs[window_length] = tf.contrib.layers.flatten(self.conv_outputs[window_length])
        # self.concatenated_output = tf.concat(list(self.flattened_convs.values()), axis=1)
        # # FC Layer 1
        # kernel_0 = tf.get_variable('kernel_0', shape=[total_dim, 1024], dtype=tf.float32)
        # bias_0 = tf.get_variable('bias_0', shape=[1024], dtype=tf.float32)
        # net = tf.matmul(self.concatenated_output, kernel_0) + bias_0
        # net = tf.nn.relu(net)
        # # net = tf.layers.dense(self.concatenated_output, 1024, activation=tf.nn.relu)
        # self.classifier_output = tf.nn.dropout(net, keep_prob=self.keep_prob)
        # print("X")

    def get_softmax_layer(self):
        num_of_classes = self.corpus.get_num_of_classes()
        # Softmax output layer
        with tf.name_scope('softmax'):
            self.logits = tf.layers.dense(self.finalFeature, num_of_classes, activation=None)
            predictions = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(predictions, 1, name='predictions')

    def train(self):
        run_id = DbLogger.get_run_id()
        explanation = DeepClassifier.get_explanation()
        DbLogger.write_into_table(rows=[(run_id, explanation)], table=DbLogger.runMetaData, col_count=2)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        trainable_var_dict = {v.name: v for v in tf.trainable_variables()}
        saver = tf.train.Saver(trainable_var_dict)
        tf.assign(self.embeddings, self.wordEmbeddings).eval(session=self.sess)
        losses = []
        iteration = 0
        for epoch_id in range(GlobalConstants.EPOCH_COUNT_CLASSIFIER):
            print("*************Epoch {0}*************".format(epoch_id))
            self.corpus.set_current_dataset_type(dataset_type=DatasetType.Training)
            while True:
                data, labels, lengths, document_ids, max_length = \
                    self.corpus.get_next_training_batch(batch_size=GlobalConstants.BATCH_SIZE)
                feed_dict = {self.batch_size: GlobalConstants.BATCH_SIZE,
                             self.input_word_codes: data,
                             self.input_y: labels,
                             self.keep_prob: GlobalConstants.DROPOUT_KEEP_PROB,
                             self.sequence_length: lengths,
                             self.isTrainingFlag: True,
                             self.max_sequence_length: max_length}
                # run_ops = [self.inputs, self.conv_outputs, self.flattened_convs, self.concatenated_output]
                run_ops = [self.optimizer, self.mainLoss, self.conv_outputs, self.padded_conv_outputs,
                           self.embeddingImage]
                results = self.sess.run(run_ops, feed_dict=feed_dict)
                losses.append(results[1])
                iteration += 1
                if iteration % 100 == 0:
                    avg_loss = np.mean(np.array(losses))
                    print("Iteration:{0} Avg. Loss:{1}".format(iteration, avg_loss))
                    losses = []
                if self.corpus.isNewEpoch:
                    # Save the model
                    path = os.path.join(GlobalConstants.CNN_MODEL_CHECKPOINT_PATH,
                                        "cnn{0}_epoch{1}.ckpt".format(run_id, epoch_id))
                    saver.save(self.sess, path)
                    print("Original results")
                    training_accuracy, doc_training_accuracy = self.test(dataset_type=DatasetType.Training)
                    test_accuracy, doc_test_accuracy = self.test(dataset_type=DatasetType.Validation)

                    DbLogger.write_into_table(rows=[(run_id, epoch_id, training_accuracy, test_accuracy,
                                                     doc_training_accuracy, doc_test_accuracy)],
                                              table=DbLogger.logsTable, col_count=6)
                    break
