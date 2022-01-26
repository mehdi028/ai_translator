import tensorflow as tf
from tensorflow.contrib import lookup, rnn, layers, seq2seq
from tensorflow import nn
import pickle
import numpy as np


class DataFlow(object):

    def __init__(self, data_filename, src_words_filename, target_words_filename, percent,
                 batch_size, shuffle=False):
        self.data_filename = data_filename
        self.batch_size = batch_size
        self.source_word2int = lookup.index_table_from_file(src_words_filename, num_oov_buckets=1)
        self.src_words_filename = src_words_filename
        self.target_word2int = lookup.index_table_from_file(target_words_filename, num_oov_buckets=1)
        self.target_words_filename = target_words_filename
        self.source_int2word = lookup.index_to_string_table_from_file(src_words_filename)
        self.target_int2word = lookup.index_to_string_table_from_file(target_words_filename)
        self._source_data, self._target_data = self._read_binary_data(data_filename)
        self._src_train_data, self._target_train_data, self._src_val_data, self._target_val_data\
            = self._split_data(percent, shuffle)
        self.nb_train_batches = len(self._src_train_data) // self.batch_size
        self.nb_val_batches = len(self._src_val_data) // self.batch_size
        self.train_dataset = self._prepare_zip_data(self._src_train_data, self._target_train_data)

        self.validation_dataset = self._prepare_zip_data(self._src_val_data, self._target_val_data)
        # number of tokens attributes
        self.source_num_tokens = self._count_words(src_words_filename) + 1
        # print('object source words {}'.format(self.source_num_tokens)) #TODO DELETE THIS LINE !
        self.target_num_tokens = self._count_words(target_words_filename) + 1
        # print('object target num words tokens {}'.format(self.target_num_tokens))  # TODO DELETE THIS LINE !!
        print('Dataflow object has been created successfully !')


    @staticmethod
    def _read_binary_data(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def _split_data(self, percent, shuffle=False):
        ''' split data into train and validation data

        percent(float): the percentage of the train data

        '''
        source_data = self._source_data
        target_data = self._target_data
        if shuffle:
            source_data = np.array(self._source_data)
            target_data = np.array(self._target_data)
            indices = np.arange(len(self._source_data))
            np.random.shuffle(indices)
            source_data = source_data[indices]
            target_data = target_data[indices]
        source_train_data = source_data[: int(len(self._source_data) * percent)]
        target_train_data = target_data[: int(len(self._target_data) * percent)]
        source_val_data = source_data[int(len(self._source_data) * percent):]
        target_val_data = target_data[: int(len(self._target_data) * percent)]

        return source_train_data, target_train_data, source_val_data, target_val_data

    def _prepare_zip_data(self, source_data, target_data):
        source_dataset = tf.data.Dataset.from_tensor_slices(source_data)
        target_dataset = tf.data.Dataset.from_tensor_slices(target_data)
        source_dataset = source_dataset.map(lambda strings: tf.string_split([strings]).values)

        target_dataset = target_dataset.map(lambda strings: tf.string_split([strings]).values)
        source_dataset = source_dataset.map(lambda words: (self.source_word2int.lookup(words), tf.size(words)))
        target_dataset = target_dataset.map(lambda words: (self.target_word2int.lookup(words), tf.size(words)))
        return tf.data.Dataset.zip((source_dataset, target_dataset))

    def batch_train_dataset(self, batch, drop=False):
        padded_shape = (tf.TensorShape([None]), tf.TensorShape([])),\
                       (tf.TensorShape([None]), tf.TensorShape([]))
        padding_value = (self.source_word2int.lookup(tf.constant('<PAD>')), 0),\
                        (self.target_word2int.lookup(tf.constant('<PAD>')), 0)
        return self.train_dataset.padded_batch(batch, padded_shape, padding_value, drop_remainder=drop).repeat()

    def batch_validation_dataset(self, batch, drop=False):
        padded_shape = (tf.TensorShape([None]), tf.TensorShape([])),\
                       (tf.TensorShape([None]), tf.TensorShape([]))
        padding_value = (self.source_word2int.lookup(tf.constant('<PAD>')), 0),\
                        (self.target_word2int.lookup(tf.constant('<PAD>')), 0)
        return self.validation_dataset.padded_batch(batch, padded_shape, padding_value, drop_remainder=drop).repeat()

    def count_words(self, src=True):
        if src:
            f = open(self.src_words_filename, 'r')
            lines = f.readlines()
            f.close()
            return len(lines)
        else:
            f = open(self.target_words_filename, 'r')
            lines = f.readlines()
            f.close()
            return len(lines)

    @staticmethod
    def _count_words(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        return len(lines)



class Seq2Seq(object):

    def __init__(self, dataflow: DataFlow, rnn_size, keep_prob, lr, layers, enc_embd_dim, dec_embd_dim,
                 batchsize, mode='train', biderectional=True, att_mechanism=None, beam_search=False, width=None):
        self._data_flow = dataflow
        # self.train_data = self._data_flow.batch_train_dataset(self.batch_size)
        # self.val_data = self._data_flow.batch_validation_dataset(self.batch_size)
        self._rnn_size = rnn_size
        self._keep_prob = keep_prob
        self._lr = lr
        self._layers = layers
        self._enc_embd_dim = enc_embd_dim
        self._dec_embd_dim = dec_embd_dim
        self.batch_size = batchsize
        self._bidirectional = biderectional
        self.mode = mode
        self._beam_search = beam_search
        self.att_mechanism = att_mechanism
        if beam_search:

            self._width = width

    def prepare__train_val_data(self):
        train_dataset = self._data_flow.batch_train_dataset(self.batch_size)
        val_dataset = self._data_flow.batch_validation_dataset(self.batch_size)

        return train_dataset, val_dataset

    def model(self, src_input, src_lengths, target_input, target_lengths):
        enc_output, enc_state = self.encoder(src_input, src_lengths)
        # preprocess target
        dec_target = self._preprocess_targets(target_input)
        # decoder
        logits, predictions = self.decoder(enc_output, enc_state, src_lengths, dec_target, target_lengths)
        return logits, predictions

    def _dropout_cell(self):
        cell = rnn.LSTMBlockCell(self._rnn_size)
        wrap_cell = rnn.DropoutWrapper(cell, self._keep_prob)
        return wrap_cell

    def encoder(self, source_inputs, source_seq_lengths):
        ''' encode the source sentence
        source_inputs : ids from the source data
        '''
        nb_layers = self._layers

        enc_tokens = self._data_flow.source_word2int.size()
        # preapare the embedding layer and the rnn_layers
        enc_embeding = layers.embed_sequence(source_inputs, enc_tokens, self._enc_embd_dim)
        if self._bidirectional:
            nb_layers -= 2
            fw_cell = self._dropout_cell()
            bw_cell = self._dropout_cell()
            # bioutput is a tuple if forward and backward  cells
            bi_output, _ = nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, enc_embeding)
            # input for the  unidirecctional rnn
            enc_embeding = tf.concat(bi_output, -1)
        nb_layers = [self._dropout_cell() for _ in range(nb_layers)]
        stacked_layers = rnn.MultiRNNCell(nb_layers)
        enc_output, enc_state = nn.dynamic_rnn(stacked_layers, enc_embeding, source_seq_lengths, tf.float32)
        return enc_output, enc_state

    # preprocess the target_sentences
    def _preprocess_targets(self, targets):
        strided_tensor = tf.strided_slice(targets, (0, 0), (self.batch_size, -1), (1, 1))
        start_value = self._data_flow.target_word2int.lookup('<GO>')
        start_tensor = tf.fill([self.batch_size, 1], start_value)
        concat = tf.concat((start_tensor, strided_tensor), -1)
        return concat

    def attention_mechanism(self, dec_layers, encoder_output, encoder_state, source_seq_lengths, batchsize):
        if not self._bidirectional:
            attention = seq2seq.LuongAttention(self._rnn_size, encoder_output, source_seq_lengths)
        else:
            attention = seq2seq.BahdanauAttention(self._rnn_size, encoder_output, source_seq_lengths)

        # dec_layers = rnn.MultiRNNCell([self._dropout_cell() for _ in range(self._layers)])
        decoder_rnn = seq2seq.AttentionWrapper(dec_layers, attention)
        initial_state = decoder_rnn.zero_state(batchsize, tf.float32)
        initial_state = initial_state.clone(cell_state=encoder_state)
        return decoder_rnn, initial_state

    def training_layer(self, projection_layer, embedding_lookup, dec_init_state, dec_rnn, target_seq_len, scope):
        helper = seq2seq.TrainingHelper(embedding_lookup, target_seq_len, name='helper')
        # basic decoder
        training_decoder = seq2seq.BasicDecoder(dec_rnn, helper, dec_init_state, projection_layer)
        decoder_logits, _, _ = seq2seq.dynamic_decode(training_decoder, impute_finished=True, scope=scope)
        return tf.identity(decoder_logits.rnn_output, name='logits')

    def inference_layer(self, projection_layer, embedding_layer, dec_init_state, dec_rnn,
                        source_seq_len, batchsize, scope):
        # start_vector = tf.fill([batchsize], self._data_flow.target_table.lookup('<GO>'))
        start_vector = tf.tile(tf.constant(self._data_flow.target_word2int.lookup('<GO>'), dtype=tf.int32), [batchsize])
        end_token = self._data_flow.target_word2int.lookup('<EOS>')
        helper = seq2seq.GreedyEmbeddingHelper(embedding_layer,
                                               start_vector,
                                               end_token)
        max_iter = tf.round(tf.reduce_max(source_seq_len) * 2)
        if not self._beam_search:
            infer_decoder = seq2seq.BasicDecoder(dec_rnn, helper, dec_init_state, projection_layer)
            decoder_predictions, _, _ = seq2seq.dynamic_decode(infer_decoder,
                                                               impute_finished=True,
                                                               maximum_iterations=max_iter,
                                                               scope=scope)

            predictions = tf.identity(decoder_predictions.sample_id, name='predictions')

        else:
            infer_decoder = seq2seq.BeamSearchDecoder(dec_rnn,
                                                      embedding_layer,
                                                      start_vector,
                                                      end_token,
                                                      dec_init_state,
                                                      beam_width=self._width,
                                                      output_layer=projection_layer)
            decoder_predictions, _, _ = seq2seq.dynamic_decode(infer_decoder,
                                                               impute_finished=True,
                                                               maximum_iterations=max_iter,
                                                               scope=scope)
            predictions = tf.identity(decoder_predictions.predected_ids, name='predictions')

        return predictions

    def decoder(self, encoder_output, encoder_state, source_seq_lengths, ids, target_seq_lengths):
        # embedding layer
        target_tokens = self._data_flow.target_word2int.size()
        embedding_layer = tf.get_variable('dec_embedding', (target_tokens, self._dec_embd_dim))
        embedding_lookup = nn.embedding_lookup(embedding_layer, ids)
        # projection layer
        projection_layer = tf.layers.Dense(target_tokens)
        # setup the decoder initial state and the decoder rnn in different circumstances
        dec_layers = rnn.MultiRNNCell([self._dropout_cell() for _ in range(self._layers)])
        encoder_state = [encoder_state[-1] for _ in range(self._layers)]
        batchsize = self.batch_size
        if self.mode == 'infer' and self._beam_search:
            encoder_output = seq2seq.tile_batch(encoder_output, self._width)
            encoder_state = seq2seq.tile_batch(encoder_state, self._width)
            source_seq_lengths = seq2seq.tile_batch(source_seq_lengths, self._width)
            batchsize = self.batch_size * self._width
        if self.att_mechanism:
            dec_layers, dec_initial_state = self.attention_mechanism(dec_layers,
                                                                     encoder_output,
                                                                     encoder_state,
                                                                     source_seq_lengths,
                                                                     batchsize)
        else:
            # dec_rnn = rnn.MultiRNNCell([self._dropout_cell() for _ in range(self._layers)])
            dec_initial_state = encoder_state

        with tf.variable_scope('decoder') as scope:
            logits = self.training_layer(projection_layer,
                                         embedding_lookup,
                                         dec_initial_state,
                                         dec_layers,
                                         target_seq_lengths,
                                         scope)
        with tf.variable_scope('decoder', reuse=True) as scope:
            predictions = self.inference_layer(projection_layer,
                                               embedding_layer,
                                               dec_initial_state,
                                               dec_layers,
                                               source_seq_lengths,
                                               batchsize, scope)

        return logits, predictions

    # optimazation process
    def optimizer(self, logits, targets, target_sequence_length):
        mask_tensor = tf.sequence_mask(target_sequence_length,
                                       tf.reduce_max(target_sequence_length),
                                       tf.float32)
        with tf.name_scope('optimization'):
            loss = seq2seq.sequence_loss(logits, targets, mask_tensor, 'loss')
            optimizer = tf.train.AdamOptimizer(self._lr)
            gradients = optimizer.compute_gradients(loss)
            clipped_grad = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(clipped_grad)
            return loss, train_op







if __name__ == '__main__':
    test_graph = tf.Graph()

    with test_graph.as_default():
        data_flow = DataFlow('words/ger_eng.p', 'words/english_words.txt', 'words/german_words.txt', 0.8)
        test_data = data_flow.batch_train_dataset(16)
#     data_flow = DataFlow('all_data.p', 'words/english_words.txt', 'words/german_words.txt', 0.8)
#     data_flow.source_word2int.size()
#     batch_size = 8
#     train_data = data_flow.batch_train_dataset(batch_size)
#     val_data = data_flow.batch_train_dataset(batch_size)
#     iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
#     next_element = iterator.get_next()
#     train_init = iterator.make_initializer(train_data)
#     val_init = iterator.make_initializer(val_data)
#
#     variable_init = tf.global_variables_initializer()
#     tables_init = tf.tables_initializer()
#     with tf.Session() as sess:
#         sess.run(variable_init)
#         sess.run(tables_init)
#         sess.run(train_init)
#         for _ in range(5):
#             print(sess.run(next_element))
#             print('-' * 20)
#

# a = ['mehdi is going home .', 'life is good .', 'bring merry home .', 'lets go .']
# a.pop()
# word = ' <EOS>'
# b = [sentence + word for sentence in a]
# print(b)
# print(a)