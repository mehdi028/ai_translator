import tensorflow as tf
from tensorflow.contrib import rnn, layers, lookup


from tensorflow.contrib import seq2seq


class Translator(object):
    def __init__(self, units, nb_layers, enc_embd_dim, dec_embd_dim, dataset_obj,
                 lr, kp, bi_dir=True, attention=True, beam_search=False, beam_width=None,
                 mode='train'):
        # the model parameters
        self._units = units
        self._nb_layers = nb_layers
        self._enc_embd_dim = enc_embd_dim
        self._dec_embd_dim = dec_embd_dim
        self.lr = lr
        self.kp = kp
        self._bi_directional = bi_dir
        self._attention = attention
        self.beam_search = beam_search
        self.beam_width = beam_width
        self._mode = mode
        # holder for the encoder output and the encoder state
        self.enc_output = None
        self.enc_state = None
        # the dataset parameters
        self.batch_size = dataset_obj.batch_size
        self.dataset = dataset_obj
        self.src_table = dataset_obj.src_table
        self.target_table = dataset_obj.target_table

    def preprocess_target(self, targets):
        strided_target = tf.strided_slice(targets, (0, 0), (self.batch_size, -1), (1, 1))
        start_value = self.target_table.lookup('<GO>')
        start_vector = tf.fill([self.batch_size, 1], start_value)
        prepared_target = tf.concat([start_vector, strided_target], axis=-1)
        return prepared_target

    # def setup_dataset(self, src_path, target_path, src_lookup_path, target_lookup_path,
    #                   shuffle=True, buffer_size=100):
    #     # load data
    #     src_data = tf.data.TextLineDataset(src_path)
    #     target_data = tf.data.TextLineDataset(target_path)
    #     # create table that holds source data and target data words
    #     self.src_table = lookup.index_table_from_file(src_lookup_path, num_oov_buckets=1)
    #     self.target_table = lookup.index_table_from_file(target_lookup_path, num_oov_buckets=1)
    #     src_data = src_data.map(lambda string: tf.string_split([string]).values)
    #     target_data = target_data.map(lambda string: tf.string_split([string]).values)
    #     # tuple  of word ids and the dynamic length
    #     src_data = src_data.map(lambda tokens: (self.src_table.lookup(tokens), tf.size(tokens)))
    #     target_data = target_data.map(lambda tokens: (self.target_table.lookup(tokens), tf.size(tokens)))
    #     # zip both source and target data
    #     dataset = tf.data.Dataset.zip((src_data, target_data))
    #     # pad the data
    #     data_shape = ((tf.TensorShape([None]), tf.TensorShape([])),
    #                   (tf.TensorShape([None]), tf.TensorShape([])))
    #
    #     data_values = (self.src_table['<PAD>'], 0), (self.target_table['<PAD>'], 0)
    #     dataset = dataset.repeat()
    #     if shuffle:
    #         dataset = dataset.shuffle(buffer_size=buffer_size)
    #
    #     self.train_dataset = dataset.padded_batch(self.batch_size, data_shape, data_values)
    #     return dataset, self.src_table, self.target_table
    
    def _rnn_cell(self):
        cell = rnn.LSTMBlockCell(self._units)
        wrap_cell = rnn.DropoutWrapper(cell, tf.constant(self.kp, tf.float16))
        return wrap_cell

    # the encoder
    def encoder(self, src_tokens, src_seq_length):

        encoder_input = layers.embed_sequence(self.batch_size, src_tokens, self._enc_embd_dim)
        nb_layers = self._nb_layers
        if self._bi_directional:
            nb_layers -= 2
            fw_cell = self._rnn_cell()
            bw_cell = self._rnn_cell()
            bi_output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, encoder_input, src_seq_length)
            encoder_input = tf.concat(bi_output, axis=-1)

        uni_cells = [self._rnn_cell() for _ in range(nb_layers)]
        stack_uni_layers = rnn.MultiRNNCell(uni_cells)
        enc_output, enc_state = tf.nn.dynamic_rnn(stack_uni_layers, encoder_input, src_seq_length)
        self.enc_output, self.enc_state = enc_output, enc_state
        return enc_output, enc_state

    def _attention_mechanism(self, enc_out, enc_state, src_sequence_length, dec_cells):
        # dec_cells = [self._rnn_cell() for _ in range(self._nb_layers)]
        if self._bi_directional:
            mechanism = seq2seq.BahdanauAttention(self._units, enc_out, src_sequence_length)
        else:
            mechanism = seq2seq.LuongAttention(self._units, enc_out, src_sequence_length)

        decoder_rnn = seq2seq.AttentionWrapper(dec_cells, mechanism)
        # we need the encoder last state  nb layers time
        last_states = tuple(enc_state[-1] for _ in range (self._nb_layers))
        if self.beam_search and self._mode == 'infer':
            dec_state = decoder_rnn.zero_state(self.batch_size * self.beam_width, tf.float16)
        else:
            dec_state = decoder_rnn.zero_state(self.batch_size, tf.float16)

        dec_state = dec_state.clone(cell_state=last_states)
        return decoder_rnn, dec_state

    def decoder(self, targets, src_sequence_lengths, dec_tokens, target_seq_lengths):
        dec_layer_cells = [self._rnn_cell() for _ in range(self._nb_layers)]
        decoder_cells = rnn.MultiRNNCell(dec_layer_cells)

        if self.beam_search and self._mode == 'infer':
            encoder_output = seq2seq.tile_batch(self.enc_output, self.beam_width)
            decoder_state = seq2seq.tile_batch(self.enc_output, self.beam_width)
            src_sequence_lengths = seq2seq.tile_batch(src_sequence_lengths, self.beam_width)
        else:
            encoder_output = self.enc_output
            decoder_state = self.enc_state

        if self._attention:
            decoder_cells, decoder_state = self._attention_mechanism(encoder_output, decoder_state,
                                                                     src_sequence_lengths, decoder_cells)

        embedding_layer = tf.get_variable('dec_embedding', (dec_tokens, self._dec_embd_dim))
        embedding_lookup = tf.nn.embedding_lookup(embedding_layer, targets)
        projection_layer = tf.layers.Dense(dec_tokens)

        with tf.variable_scope('decode'):
            # trainghelper
            helper = seq2seq.TrainingHelper(embedding_lookup, target_seq_lengths, name='training_helper')
            # basic decoder
            decoder = seq2seq.BasicDecoder(decoder_cells,
                                           helper,
                                           decoder_state,
                                           projection_layer)
            # the output
            decode, _ = seq2seq.dynamic_decode(decoder)
            logits = tf.identity(decode.rnn_output, name='logits')

        with tf.variable_scope('decode', reuse=True):
            start_vector = tf.fill([self.batch_size], self.target_table.lookup('<GO>'), name='start_vector')
            max_iteration = tf.round(tf.reduce_max(src_sequence_lengths)) * 2
            # infer_helper = seq2seq.GreedyEmbeddingHelper(embedding_layer, start_vector, target_vocab2int['<EOS>'])
            # decoder for both  basic and beam_search
            if self._mode == 'infer' and self.beam_search:
                beam_decoder = seq2seq.BeamSearchDecoder(decoder_cells,
                                                         embedding_layer,
                                                         start_vector,
                                                         self.target_table('<EOS>'),
                                                         decoder_state,
                                                         self.beam_width,
                                                         projection_layer)

                # the output
                decode_predictions = seq2seq.dynamic_decode(beam_decoder, maximum_iterations=max_iteration)
                predictions = tf.identity(decode_predictions.sample_id, name='beam_predictions')
            else:
                # helper
                infer_helper = seq2seq.GreedyEmbeddingHelper(embedding_layer,
                                                             start_vector,
                                                             self.target_table.lookup('<EOS>'))
                # the output
                infer_decoder = seq2seq.BasicDecoder(decoder_cells,
                                                     infer_helper,
                                                     decoder_state,
                                                     projection_layer)

                # the output
                decode_predictions = seq2seq.dynamic_decode(infer_decoder, maximum_iterations=max_iteration)
                predictions = tf.identity(decode_predictions.sample_id, name='basic_predictions')

        return logits, predictions


# def setup_dataset(src_path, target_path, src_lookup_path, target_lookup_path,
#                   batch_size, shuffle=True, buffer_size=100):
#     # load data
#     src_data = tf.data.TextLineDataset(src_path)
#     target_data = tf.data.TextLineDataset(target_path)
#     # create table that holds source data and target data words
#     src_table = lookup.index_table_from_file(src_lookup_path, num_oov_buckets=1)
#     target_table = lookup.index_table_from_file(target_lookup_path, num_oov_buckets=1)
#     src_data = src_data.map(lambda string: tf.string_split([string]).values)
#     target_data = target_data.map(lambda string: tf.string_split([string]).values)
#     # tuple  of word ids and the dynamic length
#     src_data = src_data.map(lambda tokens: (src_table.lookup(tokens), tf.size(tokens)))
#     target_data = target_data.map(lambda tokens: (target_table.lookup(tokens), tf.size(tokens)))
#     # zip both source and target data
#     dataset = tf.data.Dataset.zip((src_data, target_data))
#     # pad the data
#     data_shape = ((tf.TensorShape([None]), tf.TensorShape([])),
#                   (tf.TensorShape([None]), tf.TensorShape([])))
#
#     data_values = (src_table['<PAD>'], 0), (target_table['<PAD>'], 0)
#     dataset = dataset.repeat()
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=buffer_size)
#
#     dataset = dataset.padded_batch(batch_size, data_shape, data_values)
#     return dataset, src_table, target_table
    

class DataSet(object):
    def __init__(self, src_path, target_path, src_lookup_path, target_lookup_path, batch_size, src_val_path=None,
                 target_val_path=None, shuffle=True, buffer_size=100):

        # self.src_data = tf.data.TextLineDataset(src_path)
        # self.target_data = tf.data.TextLineDataset(target_path)
        self.src_table = lookup.index_table_from_file(src_lookup_path, num_oov_buckets=1)
        self.target_table = lookup.index_table_from_file(target_lookup_path, num_oov_buckets=1)
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_data = self.setup_data(src_path, target_path)
        if src_val_path:
            self.val_data = self.setup_data(src_val_path, target_val_path)

    def setup_data(self, src_data_path, target_data_path):
        src_data = tf.data.TextLineDataset(src_data_path)
        target_data = tf.data.TextLineDataset(target_data_path)

        # create table that holds source data and target data words
        # src_table = lookup.index_table_from_file(self.src_lookup_path, num_oov_buckets=1)
        # target_table = lookup.index_table_from_file(self.target_lookup_path, num_oov_buckets=1)
        src_data = src_data.map(lambda string: tf.string_split([string]).values)
        target_data = target_data.map(lambda string: tf.string_split([string]).values)
        # tuple  of word ids and the dynamic length
        src_data = src_data.map(lambda tokens: (self.src_table.lookup(tokens), tf.size(tokens)))
        target_data = target_data.map(lambda tokens: (self.target_table.lookup(tokens), tf.size(tokens)))
        # zip both source and target data
        dataset = tf.data.Dataset.zip((src_data, target_data))
        # pad the data
        data_shape = ((tf.TensorShape([None]), tf.TensorShape([])),
                      (tf.TensorShape([None]), tf.TensorShape([])))

        data_values = (self.src_table.lookup(tf.constant('<PAD>')), 0),\
                      (self.target_table.lookup(tf.constant('<PAD>')), 0)
        dataset = dataset.repeat()
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        dataset = dataset.padded_batch(self.batch_size, data_shape, data_values)
        return dataset
