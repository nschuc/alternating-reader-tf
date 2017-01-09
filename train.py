import tensorflow as tf
from datetime import datetime
import os
from collections import defaultdict
import numpy as np

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

    for x, q, y in get_batch(X, Q, Y, config.batch_size):
        batch_loss, summary, attentions =  model.batch_predict(x, q, y)
        total_accuracy += compute_accuracy(x, attentions, y)
        total_loss += batch_loss
        batch_num += 1
    total_accuracy /= batch_num
    total_loss /= batch_num
    return total_loss, total_accuracy

def run(config, sess, model, train_data, test_data):
    X_train, Q_train, Y_train = train_data
    X_test, Q_test, Y_test = test_data

    timestamp = str(datetime.now())
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    config.log_dir = os.path.join(config.log_dir, timestamp)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'test'))


    learning_rate = config.learning_rate
    last_accuracy = 0
    valid_acc = 0
    half_epoch = 500 * (len(X_train) / (2 * config.batch_size) // 500)

    for epoch in range(config.num_epochs):
        # Train over epoch
        for X, Q, Y in get_batch(X_train, Q_train, Y_train, config.batch_size):
            batch_loss, summary, step, attentions = model.batch_fit(X, Q, Y, learning_rate)
            train_writer.add_summary(summary, step)
            train_accuracy = compute_accuracy(X, attentions, Y)
            print('Step {}: Train batch (loss, acc): ({},{})'.format(step, batch_loss, train_accuracy))
            if step % config.evaluate_every == 0:
                batch = random_batch(X_test, Q_test, Y_test, config.batch_size)
                test_loss, summary, attentions = model.batch_predict(*batch)
                accuracy = compute_accuracy(batch[0], attentions, batch[2])
                last_accuracy = accuracy
                test_writer.add_summary(summary, step)
                print('Step {}: Test batch (loss, acc): ({},{})'.format(step, test_loss, accuracy))
            if step % half_epoch == 0:
                # Validation loss after epoch
                valid_loss, new_valid_acc = run_epoch(model, X_test, Q_test, Y_test)
                if new_valid_acc >= valid_acc:
                    learning_rate = learning_rate * config.learning_rate_decay
                    print("Decaying learning rate to", learning_rate)
                valid_acc = new_valid_acc
                print('Epoch {} - validation loss: {}, accuracy: {}'.format(epoch, valid_loss, valid_acc))
                path = saver.save(sess, checkpoint_prefix + '_{:.3f}_{:.3f}'.format(valid_loss, valid_acc), global_step=step)
                print("Saved model checkpoint to {}\n".format(path))

