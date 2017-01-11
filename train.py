import tensorflow as tf
from collections import defaultdict
import numpy as np
import os
from tqdm import *

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

def run_epoch(config, model, X, Q, Y):
    batch_num = 0
    total_loss = 0
    total_accuracy = 0

    for x, q, y in get_batch(X, Q, Y, config.batch_size):
        batch_loss, summary, attentions =  model.batch_predict(x, q, y)
        total_accuracy += compute_accuracy(x, attentions, y)
        total_loss += batch_loss
        batch_num += 1
    total_accuracy /= batch_num
    total_loss /= batch_num
    return total_loss, total_accuracy

def trace(config, sess, model, train_data):
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    X, Q, Y = random_batch(*train_data, config.batch_size)
    model.batch_fit(X, Q, Y, learning_rate, run_options, run_metadata)
    train_writer.add_run_metadata(run_metadata, 'step%d' % step)

    from tensorflow.python.client import timeline
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)
    return

def run(config, sess, model, train_data, test_data, saver=None):
    X_train, Q_train, Y_train = train_data
    X_test, Q_test, Y_test = test_data

    train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'test'))

    learning_rate = config.learning_rate
    last_accuracy = 0
    valid_acc = 0
    half_epoch = 500 * (len(X_train) / (2 * config.batch_size) // 500)

    if config.restore_file is not None:
        print('[!] Loading variables from checkpoint %s' % config.restore_file)
        saver.restore(sess, config.restore_file)

    num_batches = len(X_train) - len(X_train) % config.batch_size
    for epoch in range(config.num_epochs):
        for start in tqdm(range(0, num_batches, config.batch_size)):
            end = start + config.batch_size
            X, Q, Y = (X_train[start:end], Q_train[start:end], Y_train[start:end])
            batch_loss, summary, step, attentions = model.batch_fit(
                    X, Q, Y, learning_rate)
            train_writer.add_summary(summary, step)
            if step % config.evaluate_every == 0:
                print('[!] Running batch on validation set for evaluation')
                X, Q, Y = random_batch(X_test, Q_test, Y_test, config.batch_size)
                test_loss, summary, attentions = model.batch_predict(X, Q, Y)
                accuracy = compute_accuracy(X, attentions, Y)
                last_accuracy = accuracy
                test_writer.add_summary(summary, step)
            if step % config.checkpoint_every == 0:
                print('[!] Running epoch on validation set for checkpoint')
                loss, acc = run_epoch(config, model, X_test, Q_test, Y_test)
                ckpt_file = 'model-l{:.3f}_a{:.3f}.ckpt'.format(loss, acc)
                path = saver.save(sess, os.path.join(config.ckpt_dir, ckpt_file), global_step=step)
                print('[!] Saved checkpoint to %s' % path)
            if step % half_epoch == 0: # Compute loss over validation set
                valid_loss, new_valid_acc = run_epoch(config, model, X_test, Q_test, Y_test)
                if new_valid_acc >= valid_acc:
                    learning_rate = learning_rate * config.learning_rate_decay
                valid_acc = new_valid_acc
