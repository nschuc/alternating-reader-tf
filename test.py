import tensorflow as tf
from collections import defaultdict
import numpy as np
import os
from tqdm import *

def get_batch(X, Q, Y, batch_size):
    for start in range(0, len(X) - len(X) % batch_size, batch_size):
        end = start + batch_size
        yield (X[start:end], Q[start:end], Y[start:end])

def max_probability(words, probabilities):
    probs = defaultdict(int)
    for idx, word in enumerate(words):
        probs[word] += probabilities[idx] # Sum probabilities over word index
    return max(probs, key=probs.get)

def compute_accuracy(docs, probabilities, labels):
    correct_count = 0
    for doc in range(docs.shape[0]):
        guess = max_probability(docs[doc, :], probabilities[doc, :])
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

def idx2string(idx2word, words):
    return ' '.join([ idx2word[x] for x in words if x > 0])

def run(config, sess, model, test_data, word2idx, print_samples=True):
    X, Q, Y = test_data
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    test_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'test'))
    batch_size = config.batch_size
    batch_num = 0
    total_loss = 0
    total_accuracy = 0
    print('[!] Running epoch on test set')
    for start in range(0, len(X) - len(X) % batch_size, batch_size):
        end = start + batch_size
        x, q, y = (X[start:end], Q[start:end], Y[start:end])
        batch_loss, summary, attentions =  model.batch_predict(x, q, y)
        batch_accuracy = compute_accuracy(x, attentions, y)

        if print_samples:
            guess = max_probability(x[0], attentions[0])
            print('Document: {}'.format(idx2string(idx2word, x[0,:])))
            print('Query: {}'.format(idx2string(idx2word, q[0,:])))
            print('Answer: {}'.format(idx2string(idx2word,[y[0]])))
            print('Prediction: {}'.format(idx2string(idx2word, [guess])))

        print('[!] batch loss: {}, Test accuracy: {}'.format(batch_loss, batch_accuracy))

        total_loss += batch_loss
        total_accuracy += batch_accuracy
        batch_num += 1
        test_writer.add_summary(summary, batch_num)

    total_accuracy /= batch_num
    total_loss /= batch_num
    print('[!] Test loss: {}, Test accuracy: {}'.format(total_loss, total_accuracy))
