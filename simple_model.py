import tensorflow as tf
from tensorflow.contrib import seq2seq, layers, lookup, rnn
from tensorflow.python.layers.core import Dense
from robot import DataFlow


class BasicModel(object):
    def __init__(self, data_flow: DataFlow, units_size, lr_rate, keep_prob, batch_size, nb_layers, enc_embedding_dim,
                 dec_embedding_dim, drop_remain=False):
        self._nb_layers = nb_layers
        self._batch_size = batch_size
        self._units_size = units_size
        self._data_flow = data_flow
        self._src_word2int = self._data_flow.source_word2int
        self._target_word2int = self._data_flow.target_word2int
        self._lr_rate = lr_rate
        self._keep_prob = tf.constant(keep_prob, dtype=tf.float32, name='keep_prob')
        self._enc_embedding_dim = enc_embedding_dim
        self._dec_embedding_dim = dec_embedding_dim
        self._drop_remain = drop_remain

    # @staticmethod
    # def _hyperparms():
    #     lr_rate = tf.placeholder(tf.float32, (), 'learning_rate')
    #     keep_prob = tf.placeholder(tf.float32, (), 'keep_prob')
    #     return lr_rate, keep_prob
    #
    # prepare trainig and validation data
    def prepare_train_validation_data(self):
        return self._data_flow.batch_train_dataset(self._batch_size, drop=self._drop_remain), \
               self._data_flow.batch_validation_dataset(self._batch_size, self._drop_remain)

    def _rnn_cell(self):
        rnn_cell = rnn.BasicLSTMCell(self._units_size)

        wrapp_cell = rnn.DropoutWrapper(rnn_cell, self._keep_prob)

        return wrapp_cell

    # preprocess the targets
    def _preprocess_target(self, targets):
        strided_targets = tf.strided_slice(targets, (0, 0), (self._batch_size, -1), strides=(1, 1))
        start_value = self._target_word2int.lookup(tf.constant('<GO>'))
        start_matrix = tf.fill([self._batch_size, 1], start_value)
        concat_tensor = tf.concat([start_matrix, strided_targets], 1, name='preprocessed_target')
        return concat_tensor

    # craete the encoder function
    def _encoder(self, inputs, src_seq_lengths):
        with tf.variable_scope("encoder") as encoder:
            print(' encoder number of tokens :')
            nb_tokens = self._data_flow.source_num_tokens  # TODO fix this !!
            print(' num encoder tokens: {}'.format(nb_tokens))  # TODO DELETE THIS LINE !!
            embeding = layers.embed_sequence(inputs, nb_tokens, self._enc_embedding_dim)
            create_layers = [self._rnn_cell() for _ in range(self._nb_layers)]
            stack_layers = rnn.MultiRNNCell(create_layers)
            tf.summary.histogram("encoder_weights", create_layers[0].weights)
            enc_output, enc_state = tf.nn.dynamic_rnn(stack_layers,
                                                      embeding,
                                                      src_seq_lengths,
                                                      dtype=tf.float32,
                                                      scope=encoder)
            # print('the flow  passed successfully through the encoder !')  # TODO delete this line !!
            return enc_output, enc_state

    # create attention mechanism
    def attention_mechanism(self, encoder_output, encoder_state, source_seq_len):
        with tf.variable_scope("attention_block"):
            luong_attention = seq2seq.LuongAttention(self._units_size, encoder_output, source_seq_len)
            create_layers = [self._rnn_cell() for _ in range(self._nb_layers)]

            attention_layers = rnn.MultiRNNCell(create_layers)
            tf.summary.histogram("att_frist_layer", create_layers[0].weights)
            dec_cells = seq2seq.AttentionWrapper(attention_layers, luong_attention, name='attention_mechanism')
            dec_states = tuple(encoder_state[-1] for _ in range(self._nb_layers))
            zero_state = dec_cells.zero_state(self._batch_size, dtype=tf.float32)
            dec_initial_state = zero_state.clone(cell_state=dec_states)
            # print(' the flow passed successfully through the attention mechanism')  # TODO delete this line !!
            return dec_cells, dec_initial_state

    # create decoder training phase
    def _decoder_train(self, dec_cells, dec_initial_state, projection_layer, embedding_lookup,
                       target_seq_lengths):
        print(' we are in the training phase !')  # TODO Delete this line !!!
        # helper used only during training phase
        helper = seq2seq.TrainingHelper(embedding_lookup, target_seq_lengths, name='helper')
        # decoder
        decoder = seq2seq.BasicDecoder(dec_cells, helper, dec_initial_state, projection_layer)
        # dynamic decoding
        max_iter = tf.reduce_max(target_seq_lengths)
        decoder_output, _, _ = seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_iter)

        logits = tf.identity(decoder_output.rnn_output, name='logits')
        print('congratulations the decoder train phase passed the test !!')  # TODO delete this line !!!

        return logits

    def _decoder_inference(self, dec_cells, dec_initial_state, projection_layer, embedding_layer,
                           source_seq_lengths):
        # helper
        # tf. cast used to convert the dtype of the start value since it's tf.int64, and we want tf.int32
        start_value = tf.cast(self._target_word2int.lookup(tf.constant('<GO>')), tf.int32)
        start_vector = tf.fill([self._batch_size], start_value)

        helper = seq2seq.GreedyEmbeddingHelper(embedding_layer,
                                               start_vector,
                                               tf.cast(self._target_word2int.lookup(tf.constant('<EOS>')), tf.int32))

        # decoder
        decoder = seq2seq.BasicDecoder(dec_cells, helper, dec_initial_state, projection_layer)
        max_iter = tf.round(tf.reduce_max(source_seq_lengths) * 2)
        # dynamic decoding
        decoder_output, _, _ = seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_iter)

        predictions = tf.identity(decoder_output.sample_id, name='predictions')
        print('congratulations the inference phase passed the test successfully !!!')  # TODO delete this line !!!
        return predictions

    def _decoder(self, encoder_output, encoder_state, source_seq_lengths,
                 target_inputs=None, target_seq_lengths=None, mode='infer'):
        nb_tokens = self._data_flow.target_num_tokens   # TODO fix this !!
        print('num decoder tokens: {}'.format(nb_tokens))  # TODO DELETE THIS LINE !!
        # the embedding
        dec_embedding = tf.get_variable('decoder_embedding', (nb_tokens, self._dec_embedding_dim))
        # embedding_lookup = tf.nn.embedding_lookup(dec_embedding, target_inputs)
        projection_layer = Dense(nb_tokens)
        # applying the attention mechanism and retrieving  the docoder cells and the decoder initial state
        dec_cells, dec_initial_state = self.attention_mechanism(encoder_output,
                                                                encoder_state,
                                                                source_seq_lengths)
        logits = None
        if mode == 'train':

            # the training phase
            with tf.variable_scope('decoder'):
                embedding_lookup = tf.nn.embedding_lookup(dec_embedding, target_inputs)
                logits = self._decoder_train(dec_cells,
                                             dec_initial_state,
                                             projection_layer,
                                             embedding_lookup,
                                             target_seq_lengths)
        print('we finished the training phase and we reached the inference phase !')  # TODO delete this line !
        # the inference phase
        with tf.variable_scope('decoder', reuse=True):  # TODO set reuse to TRUE !!
            predictions = self._decoder_inference(dec_cells,
                                                  dec_initial_state,
                                                  projection_layer,
                                                  dec_embedding,
                                                  source_seq_lengths)
        if mode == 'train':

            return logits, predictions
        elif mode == 'infer':
            return predictions

    # collect the model
    def model(self, src_sentences_ids, src_lengths, target_data=None, target_lengths=None, mode='infer'):
        enc_output, enc_state = self._encoder(src_sentences_ids, src_lengths)

        if mode == 'train':

            target_data = self._preprocess_target(target_data)

        result = self._decoder(enc_output,
                               enc_state,
                               src_lengths,
                               target_data,
                               target_lengths,
                               mode=mode)
        if mode == 'train':
            print('congratulations ! the training mode is working perfectly !!')  # TODO delete this line
            logits, predictions = result
            return logits, predictions
        else:
            return result

    # create optimizer
    def optimizer(self, logits, targets, target_lengths):
        maxlen = tf.reduce_max(target_lengths)
        mask = tf.sequence_mask(target_lengths, maxlen, dtype=tf.float32)
        loss = seq2seq.sequence_loss(logits, targets, mask, name='loss')
        tf.summary.scalar("loss_summary", loss)
        with tf.name_scope('optimazation'):
            optim = tf.train.AdamOptimizer(self._lr_rate)
            gradients = optim.compute_gradients(loss)
            clipped_grad = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gradients if grad is not None]
            train_op = optim.apply_gradients(clipped_grad)
            return loss, train_op



