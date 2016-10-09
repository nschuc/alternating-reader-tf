import tarfile
import os
import numpy as np
from functools import reduce
import itertools
import re
import h5py

data_path = 'data/'
data_filenames = {
        'train' : 'CBTest/data/cbtest_NE_train.txt',
        'test'  : 'CBTest/data/cbtest_NE_test_2500ex.txt',
        'valid' : 'CBTest/data/cbtest_NE_valid_2000ex.txt'
        }
preprocessed = 'processed.h5'

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
                  padding='pre', truncating='pre', value=0.):
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


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for s, q, a in data:
        x = [word_idx[w] for w in s]
        xq = [word_idx[w] for w in q]
        y = np.zeros(len(word_idx) + 1)
        X.append(x)
        Xq.append(xq)
        Y.append(word_idx[a])
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen),
            np.array(Y))


def load_data(dataset='train', debug=False):
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

        vocab = sorted(set(itertools.chain(*(story + q + [answer] for story, q, answer in stories))))

        print(len(vocab))
        vocab_size = len(vocab) + 1
        story_maxlen = max(map(len, (x for x, _, _ in stories)))
        query_maxlen = max(map(len, (x for _, x, _ in stories)))

        print('Vocab size:', vocab_size)
        print('{} instances:'.format(dataset), len(stories))

        word_idx = dict((w, i + 1) for i,w in enumerate(vocab))
        X, Q, Y= vectorize_stories(stories, word_idx, story_maxlen, query_maxlen)

        h5f = h5py.File(filename + '.h5', 'w')
        h5f.create_dataset('X', data=X)
        h5f.create_dataset('Q', data=Q)
        h5f.create_dataset('Y', data=Y)
        h5f.close()
    return X, Q, Y
