import tarfile
import os
import numpy as np
from functools import reduce
import itertools
import re
import h5py
import pickle

data_path = 'data/'
data_filenames = {
        'train' : 'CBTest/data/cbtest_NE_train.txt',
        'test'  : 'CBTest/data/cbtest_NE_test_2500ex.txt',
        'valid' : 'CBTest/data/cbtest_NE_valid_2000ex.txt'
        }
vocab_file = os.path.join(data_path, 'vocab.h5')

def tokenize(sentence):
    return [s.strip() for s in re.split('(\W+)+', sentence) if s.strip()]

def parse_stories(lines):
    stories = []
    story = []
    for line in lines:
        line = line.strip()
        if not line:
            story = []
        else:
            _, line = line.split(' ', 1)
            if line:
                if '\t' in line: # query line
                    q, a, _, answers = line.split('\t')
                    q = tokenize(q)
                    stories.append((story, q, a))
                else:
                    story.append(tokenize(line))
    return stories

def get_stories(story_file):
    stories = parse_stories(story_file.readlines())
    flatten = lambda story: reduce(lambda x, y: x + y, story)
    stories = [(flatten(story), q, a) for story, q, a in stories]
    return stories


# From keras.preprocessing: https://github.com/fchollet/keras/blob/master/keras/preprocessing/sequence.py
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0.):
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def vectorize_stories(data, word2idx, doc_max_len, query_max_len):
    X = []
    Xq = []
    Y = []

    for s, q, a in data:
        x = [word2idx[w] for w in s]
        xq = [word2idx[w] for w in q]
        y = np.zeros(len(word2idx) + 1)
        X.append(x)
        Xq.append(xq)
        Y.append(word2idx[a])

    X = pad_sequences(X, maxlen=doc_max_len)
    Q = pad_sequences(Xq, maxlen=query_max_len)
    return (X, Q, np.array(Y))

def build_vocab():
    if os.path.isfile(vocab_file):
        (word2idx, doc_length, query_length) = pickle.load( open( vocab_file, "rb" ) )
    else:
        stories = []
        for key, filename in data_filenames.items():
            stories = stories + get_stories(open(os.path.join(data_path, filename)))

        doc_length = max([len(s) for s, _, _ in stories])
        query_length = max([len(q) for _, q, _ in stories])

        print('Document Length: {}, Query Length: {}'.format(doc_length, query_length))
        vocab = sorted(set(itertools.chain(*(story + q + [answer] for story, q, answer in stories))))
        vocab_size = len(vocab) + 1
        print('Vocab size:', vocab_size)
        word2idx = dict((w, i + 1) for i,w in enumerate(vocab))
        pickle.dump( (word2idx, doc_length, query_length), open( vocab_file, "wb" ) )
    return (word2idx, doc_length, query_length)

def load_data(dataset='train'):
    filename = os.path.join(data_path, data_filenames[dataset])
    # Check for preprocessed data and load that instead
    if os.path.isfile(filename + '.h5'):
        h5f = h5py.File(filename + '.h5', 'r')
        X = h5f['X'][:]
        Q = h5f['Q'][:]
        Y = h5f['Y'][:]
        h5f.close()
    else:
        stories = get_stories(open(filename))

        word2idx, doc_length, query_length = build_vocab()

        X, Q, Y = vectorize_stories(stories, word2idx, doc_length, query_length)
        h5f = h5py.File(filename + '.h5', 'w')
        h5f.create_dataset('X', data=X)
        h5f.create_dataset('Q', data=Q)
        h5f.create_dataset('Y', data=Y)
        h5f.close()
    return X, Q, Y
