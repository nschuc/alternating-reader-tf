import numpy as np
import tensorflow as tf
import os
import time
from collections import defaultdict
from datetime import datetime

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
tf.flags.DEFINE_integer("evaluate_every", 300, "Evaluate model on validation set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 10000)")

tf.flags.DEFINE_boolean("debug", False, "Debug (load smaller dataset)")
tf.flags.DEFINE_boolean("trace", False, "Whether to generate a debug trace of training step")
tf.flags.DEFINE_string("trace_file", "timeline.ctf.json", "Chrome tracefile name for debugging model (default: timeline.ctf.json)")
tf.flags.DEFINE_string("log_dir", "logs", "Directory for summary logs to be written to default (./logs/)")

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

def compute_accuracy(docs, probabilities, labels):
    correct_count = 0
    for doc in range(docs.shape[0]):
        probs = defaultdict(int)
        for idx, word in enumerate(docs[doc,:]):
            probs[word] += probabilities[doc, idx]
        guess = max(probs, key=probs.get)
        if guess == labels[doc]:
            correct_count += 1
    return correct_count / docs.shape[0]

def run_epoch(model, X, Q, Y):
    batch_num = 0
    total_loss = 0
    total_accuracy = 0

    for x, q, y in get_batch(X, Q, Y, FLAGS.batch_size):
        batch_loss, summary, attentions =  model.batch_predict(x, q, y)
        total_accuracy += compute_accuracy(x, attentions, y)
        total_loss += batch_loss
        batch_num += 1
    total_accuracy /= batch_num
    total_loss /= batch_num
    return total_loss, total_accuracy


# Train Model
with tf.Session() as sess:
    model = AlternatingAttention(FLAGS.batch_size, vocab_size, doc_len, query_len, FLAGS.encoding_dim, FLAGS.embedding_dim, FLAGS.num_glimpses, session=sess)

    timestamp = str(datetime.now())
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())

    FLAGS.log_dir = os.path.join(FLAGS.log_dir, timestamp)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    train_writer = tf.train.SummaryWriter(os.path.join(FLAGS.log_dir, 'train'), sess.graph)
    test_writer = tf.train.SummaryWriter(os.path.join(FLAGS.log_dir, 'test'))

    half_epoch = 500 * (len(X_train) / (2 * FLAGS.batch_size) // 500)
    print('Half epoch', half_epoch)
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
            break
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        print('Writing tracefile to {}'.format(FLAGS.trace_file))
        trace_file = open(FLAGS.trace_file, 'w')
        trace_file.write(trace.generate_chrome_trace_format(show_dataflow=False))
        print('Done Trace')

    valid_acc = 0
    for epoch in range(FLAGS.num_epochs):
        # Train over epoch
        for X, Q, Y in get_batch(X_train, Q_train, Y_train, FLAGS.batch_size):
            batch_loss, summary, step, attentions = model.batch_fit(X, Q, Y, learning_rate)
            train_writer.add_summary(summary, step)
            train_accuracy = compute_accuracy(X, attentions, Y)
            print('Step {}: Train batch (loss, acc): ({},{})'.format(step, batch_loss, train_accuracy))
            if step % FLAGS.evaluate_every == 0:
                batch = random_batch(X_test, Q_test, Y_test, FLAGS.batch_size)
                test_loss, summary, attentions = model.batch_predict(*batch)
                accuracy = compute_accuracy(batch[0], attentions, batch[2])
                last_accuracy = accuracy
                test_writer.add_summary(summary, step)
                print('Step {}: Test batch (loss, acc): ({},{})'.format(step, test_loss, accuracy))
            if step % half_epoch == 0:
                # Validation loss after epoch
                valid_loss, new_valid_acc = run_epoch(model, X_test, Q_test, Y_test)
                if new_valid_acc >= valid_acc:
                    learning_rate = learning_rate * FLAGS.learning_rate_decay
                    print("Decaying learning rate to", learning_rate)
                valid_acc = new_valid_acc
                print('Epoch {} - validation loss: {}, accuracy: {}'.format(epoch, valid_loss, valid_acc))
                path = saver.save(sess, checkpoint_prefix + '_{:.3f}_{:.3f}'.format(valid_loss, valid_acc), global_step=step)
                print("Saved model checkpoint to {}\n".format(path))
