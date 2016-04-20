# -*- coding: utf-8 -*-
'''
custimized seq2seq(https://github.com/farizrahman4u/seq2seq)
'''
from __future__ import absolute_import

from seq2seq.layers.encoders import LSTMEncoder
from seq2seq.layers.decoders import LSTMDecoder, LSTMDecoder2, AttentionDecoder
from seq2seq.layers.bidirectional import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.core import RepeatVector, Dense, TimeDistributedDense, Dropout, Activation
from keras.models import Sequential
import theano.tensor as T

'''
Papers:
[1] Sequence to Sequence Learning with Neural Networks (http://arxiv.org/abs/1409.3215)
[2] Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (http://arxiv.org/abs/1406.1078)
[3] Neural Machine Translation by Jointly Learning to Align and Translate (http://arxiv.org/abs/1409.0473)
'''

class Seq2seqBase(Sequential):
    '''
    Abstract class for all Seq2seq models.
    '''
    wait_for_shape = False

    def add(self, layer):
        '''
        For automatic shape inference in nested models.
        '''
        self.layers.append(layer)
        n = len(self.layers)
        if self.wait_for_shape or (n == 1 and not hasattr(layer, '_input_shape')):
            self.wait_for_shape = True
        elif n > 1:
            layer.set_previous(self.layers[-2])

    def set_previous(self, layer):
        '''
        For automatic shape inference in nested models.
        '''
        self.layers[0].set_previous(layer)
        if self.wait_for_shape:
            self.wait_for_shape = False
            for i in range(1, len(self.layers)):
                self.layers[i].set_previous(self.layers[i - 1])

    def reset_states(self):
        for l in self.layers:
            if  hasattr(l, 'stateful'):
                if l.stateful:
                    l.reset_states()

class SimpleSeq2seq(Seq2seqBase):
    '''
    Simple model for sequence to sequence learning.
    The encoder encodes the input sequence to vector (called context vector)
    The decoder decoder the context vector in to a sequence of vectors.
    There is no one on one relation between the input and output sequence elements.
    The input sequence and output sequence may differ in length.
    Arguments:
    output_dim : Required output dimension.
    hidden_dim : The dimension of the internal representations of the model.
    output_length : Length of the required output sequence.
    depth : Used to create a deep Seq2seq model. For example, if depth = 3,
    there will be 3 LSTMs on the enoding side and 3 LSTMs on the
    decoding side. You can also specify depth as a tuple. For example,
    if depth = (4, 5), 4 LSTMs will be added to the encoding side and
    5 LSTMs will be added to the decoding side.
    dropout : Dropout probability in between layers.
    '''
    def __init__(self, output_dim, hidden_dim, output_length, depth=1, dropout=0.25, **kwargs):
        super(SimpleSeq2seq, self).__init__()
        if type(depth) not in [list, tuple]:
            depth = (depth, depth)
        self.encoder = LSTM(hidden_dim, **kwargs)
        self.decoder = LSTM(hidden_dim, return_sequences=True, **kwargs)
        for i in range(1, depth[0]):
            self.add(LSTM(hidden_dim, return_sequences=True, **kwargs))
            self.add(Dropout(dropout))
        self.add(self.encoder)
        self.add(Dropout(dropout))
        self.add(RepeatVector(output_length))
        self.add(self.decoder)
        for i in range(1, depth[1]):
            self.add(LSTM(hidden_dim, return_sequences=True, **kwargs))
            self.add(Dropout(dropout))
        #if depth[1] > 1:
        self.add(TimeDistributedDense(output_dim, activation='softmax'))