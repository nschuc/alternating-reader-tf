import numpy as np
import pprint
import tensorflow as tf
import os
from datetime import datetime
from train import run
from data_helper import load_data
from model import AlternatingAttention

flags = tf.app.flags;

flags.DEFINE_integer("embedding_dim", 384, "Dimensionality of character embedding (default: 384)")
flags.DEFINE_integer("encoding_dim", 128, "Dimensionality of bidirectional GRU encoding for query / document")
flags.DEFINE_integer("num_glimpses", 8, "Number of glimpse iterations during read (default: 8)")
flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.8)")
flags.DEFINE_float("l2_reg_lambda", 1e-4, "L2 regularizaion lambda (default: 0.0001)")
flags.DEFINE_float("learning_rate", 1e-3, "AdamOptimizer learning rate (default: 0.001)")
flags.DEFINE_float("learning_rate_decay", 0.8, "How much learning rate will decay after half epoch of non-decreasing loss (default: 0.8)")

# Training parameters
flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
flags.DEFINE_integer("num_epochs", 12, "Number of training epochs (default: 12)")
flags.DEFINE_integer("evaluate_every", 300, "Evaluate model on validation set after this many steps (default: 300)")
flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 10000)")

flags.DEFINE_boolean("trace", False, "Trace (load smaller dataset)")
flags.DEFINE_string("log_dir", "logs", "Directory for summary logs to be written to default (./logs/)")
flags.DEFINE_string("ckpt_dir", "ckpts", "Directory for checkpoints default (./ckpts/)")


def main(_):
    FLAGS = tf.app.flags.FLAGS
    pp = pprint.PrettyPrinter()
    FLAGS._parse_flags()
    pp.pprint(FLAGS.__flags)

    # Load Data
    X_train, Q_train, Y_train = load_data('train')
    X_test, Q_test, Y_test = load_data('valid')

    vocab_size = np.max(X_train) + 1
    doc_len = len(X_train[0])
    query_len = len(Q_train[0])

    print('Vocabulary Size:', vocab_size)
    print('Fixed Document length:', doc_len)
    print('Fixed Query length:', query_len)

    timestamp = datetime.now().strftime('%c')
    # Create directories
    FLAGS.ckpt_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    if not os.path.exists(FLAGS.ckpt_dir):
        os.makedirs(FLAGS.ckpt_dir)

    FLAGS.log_dir = os.path.join(FLAGS.log_dir, timestamp)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)


    # Train Model
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess, tf.device('/gpu:0'):
        model = AlternatingAttention(FLAGS.batch_size, vocab_size, FLAGS.encoding_dim, FLAGS.embedding_dim, FLAGS.num_glimpses, session=sess)

        run(FLAGS, sess, model,
                (X_train, Q_train, Y_train),
                (X_test, Q_test, Y_test))

if __name__ == '__main__':
    tf.app.run()
