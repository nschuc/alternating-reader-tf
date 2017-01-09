import numpy as np
import pprint
import tensorflow as tf
import os
import time
from datetime import datetime

from train import run
from load_data import load_data
from model import AlternatingAttention

flags = tf.app.flags;

flags.DEFINE_integer("embedding_dim", 384, "Dimensionality of character embedding (default: 384)")
flags.DEFINE_integer("encoding_dim", 128, "Dimensionality of bidirectional GRU encoding for query / document")
flags.DEFINE_integer("num_glimpses", 8, "Number of glimpse iterations during read (default: 8)")
flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.8)")
flags.DEFINE_float("l2_reg_lambda", 1e-4, "L2 regularizaion lambda (default: 0.0001)")
flags.DEFINE_float("learning_rate", 1e-3, "AdamOptimizer learning rate (default: 0.001)")
flags.DEFINE_float("learning_rate_decay", 0.8, "How much learning rate will decay after half epoch of non-decreasing loss (default: 0.8)")

flags.DEFINE_float("doc_len", 1500, "Document Max Length")
flags.DEFINE_float("query_len", 300, "QUery Max Length")
# Training parameters
flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
flags.DEFINE_integer("num_epochs", 12, "Number of training epochs (default: 12)")
flags.DEFINE_integer("evaluate_every", 300, "Evaluate model on validation set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 10000)")

flags.DEFINE_boolean("debug", False, "Debug (load smaller dataset)")
flags.DEFINE_string("log_dir", "logs", "Directory for summary logs to be written to default (./logs/)")

FLAGS = tf.app.flags.FLAGS
pp = pprint.PrettyPrinter()
FLAGS._parse_flags()
pp.pprint(FLAGS.__flags)

# Load Data
X_train, Q_train, Y_train = load_data('train', FLAGS.debug)
X_test, Q_test, Y_test = load_data('test', FLAGS.debug)

vocab_size = np.max(X_train) + 1

print('Vocabulary Size:', vocab_size)

# Train Model
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.device('/gpu:0'):
        model = AlternatingAttention(FLAGS.batch_size, vocab_size, FLAGS.doc_len, FLAGS.query_len, FLAGS.encoding_dim, FLAGS.embedding_dim, FLAGS.num_glimpses, session=sess)
    saver = tf.train.Saver(tf.global_variables())
    run(FLAGS, sess, model,
            (X_train, Q_train, Y_train),
            (X_test, Q_test, Y_test))
