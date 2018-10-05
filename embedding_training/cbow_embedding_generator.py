import tensorflow as tf
import numpy as np
import math
import os

from tensorflow.contrib.framework.python.framework import checkpoint_utils
from auxilliary.db_logger import DbLogger
from global_constants import GlobalConstants


class CbowEmbeddingGenerator:
    def __init__(self, corpus, context_generator):
        self.corpus = corpus
        self.contextGenerator = context_generator
        self.train_context = tf.placeholder(tf.int32, shape=[GlobalConstants.EMBEDDING_BATCH_SIZE,
                                                             2 * GlobalConstants.CBOW_WINDOW_SIZE])
        self.train_labels = tf.placeholder(tf.int32, shape=[GlobalConstants.EMBEDDING_BATCH_SIZE, 1])
        self.global_step = tf.Variable(0, trainable=False)
        # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        vocabulary_size = self.corpus.get_vocabulary_size()
        embedding_size = GlobalConstants.EMBEDDING_SIZE
        window_size = GlobalConstants.CBOW_WINDOW_SIZE
        # These are the embeddings we are going to learn.
        self.embeddings = tf.Variable(tf.random_uniform([vocabulary_size, GlobalConstants.EMBEDDING_SIZE], -1.0, 1.0),
                                      name="embeddings")
        # Softmax weights
        self.softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                               stddev=1.0 / math.sqrt(embedding_size)),
                                           name="softmax_weights")
        self.softmax_biases = tf.Variable(tf.zeros([vocabulary_size]), name="softmax_biases")

    def get_embeddings(self, sess):
        return self.embeddings.eval(session=sess)

    # def load_embeddings(self):
    #     # pretrained_var_list = checkpoint_utils.list_variables("embeddings_epoch99.ckpt")
    #     source_array = checkpoint_utils.load_variable(checkpoint_dir=GlobalConstants.EMBEDDING_CHECKPOINT_PATH,
    #                                                   name="embeddings")
    #     tf.assign(self.embeddings, source_array).eval(session=self.sess)

    def train(self):
        sess = tf.Session()
        saver = tf.train.Saver()
        vocabulary_size = self.corpus.get_vocabulary_size()
        window_size = GlobalConstants.CBOW_WINDOW_SIZE
        num_negative_sampling = GlobalConstants.NUM_NEGATIVE_SAMPLES
        epoch_count = GlobalConstants.EPOCH_COUNT
        batch_size = GlobalConstants.EMBEDDING_BATCH_SIZE

        # Build the operations
        context_embeddings = []
        for i in range(2 * window_size):
            context_embeddings.append(tf.nn.embedding_lookup(self.embeddings, self.train_context[:, i]))
        stacked_embeddings = tf.stack(axis=0, values=context_embeddings)
        averaged_embeddings = tf.reduce_mean(stacked_embeddings, axis=0, keep_dims=False)
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=self.softmax_weights, biases=self.softmax_biases,
                                       inputs=averaged_embeddings, labels=self.train_labels,
                                       num_sampled=num_negative_sampling, num_classes=vocabulary_size))
        # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=self.global_step)
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
        iteration_count = 0
        losses = []
        sess.run(tf.global_variables_initializer())
        self.contextGenerator.validate(corpus=self.corpus, embeddings=self.get_embeddings(sess=sess))
        for epoch_id in range(epoch_count):
            print("*************Epoch {0}*************".format(epoch_id))
            while True:
                context, targets = self.contextGenerator.get_next_batch(batch_size=batch_size)
                targets = np.reshape(targets, newshape=(targets.shape[0], 1))
                feed_dict = {self.train_context: context, self.train_labels: targets}
                results = sess.run([optimizer, loss], feed_dict=feed_dict)
                losses.append(results[1])
                if iteration_count % 1000 == 0:
                    print("Iteration:{0}".format(iteration_count))
                    mean_loss = np.mean(np.array(losses))
                    print("Loss:{0}".format(mean_loss))
                    losses = []
                iteration_count += 1
                if self.contextGenerator.isNewEpoch:
                    # Save embeddings to HD
                    # os.path.join("D:\\", "deep", "BDA", "Corpus", "Data", "export.json")
                    # path = os.path.join("/raid", "users", "ucbicici", "Code", "seq_classification",
                    #                          "embedding_training", "embeddings")
                    path = os.path.join(GlobalConstants.EMBEDDING_CHECKPOINT_PATH,
                                        "embedding_epoch{0}.ckpt".format(epoch_id))
                    saver.save(sess, path)
                    # embeddings_arr = self.embeddings.eval(session=sess)
                    self.contextGenerator.validate(corpus=self.corpus, embeddings=self.get_embeddings(sess=sess))
                    break
        print("X")