import tensorflow as tf
import numpy as np


def orthogonal_initializer(scale = 1.1):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        '''
        from keras https://github.com/fchollet/keras/blob/master/keras/initializations.py
        '''
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer


class AlternatingAttention(object):
    """Iterative Alternating Attention Network"""
    def __init__(self, batch_size, vocab_size, document_size, query_size, encoding_size, embedding_size,
                    num_glimpses = 8,
                    grad_norm_clip = 5.,
                    l2_reg_coef=1e-4,
                    session=tf.Session(),
                    name='AlternatingAttention'):
        """
        Creates an iterative alternating attention network as described in https://arxiv.org/abs/1606.02245
        """
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._doc_len = document_size
        self._query_len = query_size
        self._encode_size = encoding_size
        self._infer_size = 4 * encoding_size
        self._embedding_size = embedding_size
        self._num_glimpses = num_glimpses
        self._sess = session
        self._name = name

        self._build_placeholders()
        self._build_variables()

        # Regularization
        tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(l2_reg_coef), [self._embeddings])


        # Answer probability
        doc_attentions = self._inference(self._docs, self._queries)
        P_a = tf.pack([tf.reduce_sum(tf.gather(doc_attentions[i, :], tf.where(tf.equal(a, self._docs[i, :])))) for i, a in enumerate(tf.unpack(self._answers))])
        loss_op = -tf.reduce_mean(tf.log(tf.clip_by_value(P_a,1e-10,1.0)))
        self._loss_op = loss_op
        tf.scalar_summary('loss', loss_op)

        self._doc_attentions = doc_attentions

        # Optimizer and gradients
        with tf.variable_scope("optimizer"):
            self._opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            grads_and_vars = self._opt.compute_gradients(loss_op)
            capped_grads_and_vars = [(tf.clip_by_norm(g, grad_norm_clip), v) for g,v in grads_and_vars]
            self._global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), dtype=tf.int32, trainable=False)
            self._train_op = self._opt.apply_gradients(grads_and_vars, global_step=self._global_step)

        self._summary_op = tf.merge_all_summaries()

        self._sess.run(tf.initialize_all_variables())

    def _build_placeholders(self):
        """
        Adds tensorflow placeholders for inputs to the model: documents, queries, answers.
        keep_prob and learning_rate are hyperparameters that we might like to adjust while training.
        """
        self._docs = tf.placeholder(tf.int32, [self._batch_size, self._doc_len])
        self._queries = tf.placeholder(tf.int32, [self._batch_size, self._query_len])
        self._answers = tf.placeholder(tf.int32, [self._batch_size])

        self._keep_prob = tf.placeholder(tf.float32)
        self._learning_rate = tf.placeholder(tf.float32)

    def _build_variables(self):
        with tf.variable_scope(self._name, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.22, dtype=tf.float32)):
            self._embeddings = tf.get_variable("embeddings", [self._vocab_size, self._embedding_size], dtype=tf.float32)
            self._A_q = tf.get_variable("A_q", [self._batch_size, 2*self._encode_size, self._infer_size], dtype=tf.float32)
            self._a_q = tf.get_variable("a_q", [self._batch_size, 2*self._encode_size, 1], dtype=tf.float32)

            self._A_d = tf.get_variable("A_d", [self._batch_size, 2*self._encode_size, self._infer_size + 2*self._encode_size], dtype=tf.float32)
            self._a_d = tf.get_variable("a_d", [self._batch_size, 2*self._encode_size, 1], dtype=tf.float32)

            self._g_q = tf.get_variable("g_q", [self._batch_size, self._infer_size + 6 * self._encode_size, 2 * self._encode_size])
            self._g_d = tf.get_variable("g_d", [self._batch_size, self._infer_size + 6 * self._encode_size, 2 * self._encode_size])

    def _embed(self, sequence):
        """
        performs embedding lookups for every word in the sequence
        """
        with tf.variable_scope('embed'):
            embedded = tf.nn.embedding_lookup(self._embeddings, sequence)
            return embedded

    def _bidirectional_encode(self, sequence, seq_lens, size):
        """
        Encodes sequence with two GRUs, one forward, one backward, and returns the concatenation
        """
        with tf.variable_scope('encode'):
            with tf.variable_scope("_FW") as fw_scope:
                fw_encode = tf.nn.rnn_cell.GRUCell(size)
                fw_state = tf.get_variable("gru_state", [self._batch_size, fw_encode.state_size],
                        initializer=orthogonal_initializer())
                output_fw, output_state_fw = tf.nn.dynamic_rnn(
                    cell=fw_encode, inputs=sequence,
                    initial_state=fw_state, scope=fw_scope,
                    sequence_length=seq_lens, swap_memory=True)

            with tf.variable_scope("_BW") as bw_scope:
                bw_encode = tf.nn.rnn_cell.GRUCell(size)
                bw_state = tf.get_variable("gru_state", [self._batch_size, bw_encode.state_size],
                        initializer=orthogonal_initializer())
                inputs_reverse = tf.reverse_sequence(
                    input=sequence, seq_lengths=seq_lens,
                    seq_dim=1, batch_dim=0)
                tmp, output_state_bw = tf.nn.dynamic_rnn(
                    cell=bw_encode, inputs=inputs_reverse,
                    sequence_length=seq_lens, initial_state = bw_state,
                    scope=bw_scope, swap_memory=True)
                output_bw = tf.reverse_sequence(
                  input=tmp, seq_lengths=seq_lens,
                  seq_dim=1, batch_dim=0)

            outputs = (output_fw, output_bw)
            encoded = tf.concat(2, outputs)
            return encoded

    def _glimpse(self, weights, bias, encodings, inputs):
        """
        Computes glimpse over an encoding. Attention weights are computed based on the bilinear product of
        the encodings, weight matrix, and inputs.

        Returns attention weights and computed glimpse
        """
        tf.nn.dropout(weights, self._keep_prob)
        tf.nn.dropout(inputs, self._keep_prob)
        attention = tf.batch_matmul(weights, inputs) + bias
        attention = tf.squeeze(tf.batch_matmul(encodings, attention))
        attention = tf.expand_dims(tf.nn.softmax(attention), -1)
        return attention, tf.reduce_sum(attention * encodings, 1)

    def _inference(self, docs, queries):
        """
        Computes document attentions given a document batch and query batch.
        """
        with tf.variable_scope(self._name):
            # Compute document lengths / query lengths for batch
            doc_lens = tf.to_int32(tf.argmin(docs, 1))
            query_lens = tf.to_int32(tf.argmin(queries, 1))

            # Encode Document / Query
            with tf.variable_scope('docs'):
                encoded_docs = tf.nn.dropout(self._embed(docs), self._keep_prob)
                encoded_docs = self._bidirectional_encode(encoded_docs, doc_lens, self._encode_size)
            with tf.variable_scope('queries'):
                encoded_queries = tf.nn.dropout(self._embed(queries), self._keep_prob)
                encoded_queries = self._bidirectional_encode(encoded_queries, query_lens, self._encode_size)

            with tf.variable_scope('attend') as scope:
                infer_gru = tf.nn.rnn_cell.GRUCell(self._infer_size)
                infer_state = infer_gru.zero_state(self._batch_size, tf.float32)
                for iter_step in range(self._num_glimpses):
                    if iter_step > 0:
                        scope.reuse_variables()

                    # Glimpse query and document
                    _, q_glimpse = self._glimpse(self._A_q, self._a_q, encoded_queries, tf.expand_dims(infer_state, -1))
                    d_attention, d_glimpse = self._glimpse(self._A_d, self._a_d, encoded_docs, tf.concat(1, [tf.expand_dims(infer_state, -1), tf.expand_dims(q_glimpse, -1)]))

                    # Search Gates
                    gate_concat = tf.concat(1, [tf.squeeze(infer_state), q_glimpse, d_glimpse, q_glimpse * d_glimpse])
                    gate_concat = tf.expand_dims(gate_concat, 1)

                    r_d = tf.sigmoid(tf.squeeze(tf.batch_matmul(gate_concat, self._g_d)))
                    tf.nn.dropout(r_d, self._keep_prob)
                    r_q = tf.sigmoid(tf.squeeze(tf.batch_matmul(gate_concat, self._g_q)))
                    tf.nn.dropout(r_q, self._keep_prob)

                    _, infer_state = infer_gru(tf.concat(1, [r_q * q_glimpse, r_d * d_glimpse]), tf.squeeze(infer_state))
            return tf.to_float(tf.sign(docs)) * tf.squeeze(d_attention)

    def batch_fit(self, docs, queries, answers, learning_rate=1e-3, run_options=None, run_metadata=None):
        """
        Perform a batch training iteration
        """
        feed_dict = {
            self._docs: docs,
            self._queries: queries,
            self._answers: answers,
            self._keep_prob: 0.8,
            self._learning_rate: learning_rate
            }
        loss, summary, _, step = self._sess.run([self._loss_op, self._summary_op, self._train_op, self._global_step], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        return loss, summary, step

    def batch_predict(self, docs, queries, answers):
        """
        Perform batch prediction. Computes accuracy of batch predictions.
        """
        feed_dict = {
            self._docs: docs,
            self._queries: queries,
            self._answers: answers,
            self._keep_prob: 1.
            }
        loss, summary, attentions = self._sess.run([self._loss_op, self._summary_op, self._doc_attentions], feed_dict=feed_dict)
        correct = 0.
        for i in range(docs.shape[0]):
            words = np.asarray(list(set(docs[i,:])))
            accuracies = np.zeros(words.shape)
            for j,w in enumerate(words):
                accuracy = np.sum(attentions[i, docs[i,:] == w])
                accuracies[j] = accuracy
            if words[np.argmax(accuracies)] == answers[i]:
                correct += 1
        return loss, summary, correct / docs.shape[0]
