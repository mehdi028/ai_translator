import tensorflow as tf
from tensorflow.contrib import rnn


class Seq2seq(object):
    def __init__(self, units, kp, nb_layers, lr, src_data, target_data, bidirectional=True, mode='train'):
        self.units = units
        self.kp = kp
        self.nb_layers = nb_layers
        self.lr = lr
        self.src_data = src_data
        self.target_data = target_data
        self.bidirectional = bidirectional
        self.mode = mode

    def rnn_cell(self):
        cells = rnn.LSTMBlockCell(self.units)
        wrap_cell = rnn.DropoutWrapper(cells, self.kp)
        return wrap_cell
    