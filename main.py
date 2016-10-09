import numpy as np
import tensorflow as tf
import os
import time

from load_data import load_data
from model import AlternatingAttention

tf.flags.DEFINE_integer("embedding_dim", 384, "Dimensionality of character embedding (default: 384)")
tf.flags.DEFINE_integer("encoding_dim", 128, "Dimensionality of bidirectional GRU encoding for query / document")
tf.flags.DEFINE_integer("num_glimpses", 8, "Number of glimpse iterations during read (default: 8)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.8)")
tf.flags.DEFINE_float("l2_reg_lambda", 1e-4, "L2 regularizaion lambda (default: 0.0001)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "AdamOptimizer learning rate (default: 0.001)")
tf.flags.DEFINE_float("learning_rate_decay", 0.8, "How much learning rate will decay after half epoch of non-decreasing loss (default: 0.8)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 12, "Number of training epochs (default: 12)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on validation set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps (default: 10000)")

tf.flags.DEFINE_boolean("debug", False, "Debug (load smaller dataset)")
tf.flags.DEFINE_boolean("trace", False, "Whether to generate a debug trace of training step")
tf.flags.DEFINE_string("trace_file", "timeline.ctf.json", "Chrome tracefile name for debugging model (default: timeline.ctf.json)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load Data
X_train, Q_train, Y_train = load_data('train', FLAGS.debug)
X_test, Q_test, Y_test = load_data('test', FLAGS.debug)

vocab_size = np.max(X_train) + 1
doc_len = len(X_train[0])
query_len = len(Q_train[0])

print('Vocabulary Size:', vocab_size)
print('Fixed Document length:', doc_len)
print('Fixed Query length:', query_len)

def random_batch(X, Q, Y, batch_size):
    indices = np.random.choice(len(X) - len(X) % batch_size, batch_size)
    return X[indices], Q[indices], Y[indices]

def get_batch(X, Q, Y, batch_size):
    for start in range(0, len(X) - len(X) % batch_size, batch_size):
        end = start + batch_size
        yield (X[start:end], Q[start:end], Y[start:end])

# Train Model
with tf.Session() as sess:
    model = AlternatingAttention(FLAGS.batch_size, vocab_size, doc_len, query_len, FLAGS.encoding_dim, FLAGS.embedding_dim, FLAGS.num_glimpses, session=sess)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())

    train_writer = tf.train.SummaryWriter('/tmp/logs/train', sess.graph, flush_secs=25)
    test_writer = tf.train.SummaryWriter('/tmp/logs/test', flush_secs=25)

    half_epoch = len(X_train) / 2.
    learning_rate = FLAGS.learning_rate
    last_accuracy = 0

    # Perform debug trace and produce a tf Timeline for use in the chrome trace visualizer
    if FLAGS.trace:
        print('Performing full trace of model (for GPU trace libcupti.so must be on LD_LIBRARY_PATH. I found it in /usr/local/cuda/extras/CUPTI/lib64/)')
        from tensorflow.python.client import timeline
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for X, Q, Y in get_batch(X_train, Q_train, Y_train, FLAGS.batch_size):
            model.batch_fit(X, Q, Y, learning_rate, run_options=run_options, run_metadata=run_metadata)
            print(X, Q, Y)
            break
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        print('Writing tracefile to {}'.format(FLAGS.trace_file))
        trace_file = open(FLAGS.trace_file, 'w')
        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=False))
        print('Done Trace')

    for epoch in range(FLAGS.num_epochs):
        # Train over epoch
        for X, Q, Y in get_batch(X_train, Q_train, Y_train, FLAGS.batch_size):
            batch_loss, summary, step = model.batch_fit(X, Q, Y, learning_rate)
            train_writer.add_summary(summary, step)

            if step % FLAGS.evaluate_every == 0:
                batch = random_batch(X_test, Q_test, Y_test, FLAGS.batch_size)
                valid_loss, summary, accuracy = model.batch_predict(*batch)
                if step % half_epoch == 0: # this will only happen if half_epoch is a multiple of evaluate_every so kinda hacky
                    if accuracy <= last_accuracy:
                        print("No improvement in accuracy... decaying learning rate from {} to {}", learning_rate, learning_rate * FLAGS.learning_rate_decay)
                        learning_rate *= FLAGS.learning_rate_decay
                last_accuracy = accuracy
                test_writer.add_summary(summary, step)
                print('Step {} - (train_loss, valid_loss): ({}, {})'.format(step, batch_loss, valid_loss))
                print('Step {} - Accuracy: {}'.format(step, accuracy))

            if step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=step)
                print("Saved model checkpoint to {}\n".format(path))

        # Validation loss after epoch
        batch_num = 0
        loss_sum = 0
        accuracy_sum = 0
        for X, Q, Y in get_batch(X_test, Q_test, Y_test, FLAGS.batch_size):
            batch_num += 1
            batch_loss, summary, accuracy = model.batch_predict(X, Q, Y)
            loss_sum += batch_loss
            accuracy_sum += accuracy
        print('Epoch {} - validation loss: {}, accuracy: {}'.format(epoch, loss_sum / batch_num,  accuracy_sum / batch_num))
