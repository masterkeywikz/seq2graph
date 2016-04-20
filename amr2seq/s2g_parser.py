'''
parser module for sequence to graph
'''

import gflags
FLAGS=gflags.FLAGS

#gflags.DEFINE_integer('tok_min_freq',3,'min frequence for filtering rare words')
gflags.DEFINE_integer('debuglevel',0,'debug level')
gflags.DEFINE_integer('batch_size',64,'batch size for training')
gflags.DEFINE_integer('input_seq_length',200,'maximum length for input sequence')
gflags.DEFINE_integer('word_emb_dim',300,'dimension of input word embedding')
gflags.DEFINE_integer('hidden_dim',150,'dimension of hidden layer in RNN; same for encoder and decoder')
gflags.DEFINE_integer('output_seq_length',60,'maximum length for output sequence')
gflags.DEFINE_integer('epoch',10,'training epochs')

import sys,time
from util import Alphabet
from constants import UNK, WORD2VEC_EMBEDDING_PATH
from collections import defaultdict
import numpy as np
from keras.models import Sequential
#from seq2seq.models import SimpleSeq2seq
from seq2seq_util.seq2seq_models import SimpleSeq2seq

class S2GParser(object):

    def __init__(self):
        #self.token_vocab_codebook = Alphabet()
        self.graph_vocab_codebook = Alphabet()
        self.max_input_length = FLAGS.input_seq_length
        self.max_output_length = FLAGS.output_seq_length
        self.word2vec_model = None
        self.num_train_inst = 0
        
    def setup(self, tok_seq_file, graph_seq_file):
        print >> sys.stderr, 'Setting up ...'
        print >> sys.stderr, "---------------------"
        #self.token_vocab_codebook.add(UNK)
        #count = defaultdict(int)
        
        with open(tok_seq_file,'r') as tf, open(graph_seq_file, 'r') as gf:
            self.num_train_inst = 0
            for sline in tf:
                #self.max_input_length = max(self.max_input_length, len(sline.split()))
                self.num_train_inst += 1
                #for tok in sline.split():
                #    count[tok] += 1

            for gline in gf:
                #self.max_output_length = max(self.max_output_length, len(gline.split()))
                for gtok in gline.split():
                    self.graph_vocab_codebook.add(gtok)


        #for word, freq in counts.items():
        #    if freq > FLAGS.tok_min_freq:
        #        self.token_vocab_codebook.add(word)

        print >> sys.stderr, 'Maximum input length: %d' % (self.max_input_length)
        print >> sys.stderr, 'Maximum output length: %d' % (self.max_output_length)
        #print >> sys.stderr, 'Token vocab size: %d' % (self.token_vocab_codebook.size())
        print >> sys.stderr, 'Graph token vocab size: %d' % (self.graph_vocab_codebook.size())
        print >> sys.stderr, 'Total training sentence size: %d' % (self.num_train_inst)


        self._load_word_emb()
        print >> sys.stderr, 'Word vocab size (from word2vec): %d' % (len(self.word2vec_model.vocab))
        print >> sys.stderr, 'Done.'
        
    def _load_word_emb(self):
        print >> sys.stderr, "    Loading Word2vec ..."
        print >> sys.stderr, "    ---------------------"
        import gensim
        self.word2vec_model = gensim.models.Word2Vec.load_word2vec_format(WORD2VEC_EMBEDDING_PATH, binary=True)

    def get_word_emb(self, tok):
        if tok.lower() in self.word2vec_model.vocab:
            return self.word2vec_model[tok.lower()]
        return self.word2vec_model[UNK]
        
    def get_training_batch(self, tok_seq_file, graph_seq_file):
        '''
        generator for training data
        '''
        with open(tok_seq_file, 'r') as tf, open(graph_seq_file, 'r') as gf:
            X = np.zeros((FLAGS.batch_size, self.max_input_length, FLAGS.word_emb_dim), dtype=np.float32)
            Y = np.zeros((FLAGS.batch_size, self.max_output_length, self.graph_vocab_codebook.size()), dtype=bool)
            for index, line in enumerate(zip(tf,gf)):
                
                #else:
                sline, gline = line
                s_index = index % FLAGS.batch_size
                for t_index, tok in enumerate(sline.split()[:self.max_input_length]):
                    X[s_index, t_index] = self.get_word_emb(tok)

                for g_index, gtok in enumerate(gline.split()[:self.max_output_length]):
                    gtok_index = self.graph_vocab_codebook.get_index(gtok)
                    Y[s_index, g_index, gtok_index] = 1


                if (index + 1) % FLAGS.batch_size == 0:
                    yield X, Y
                    X = np.zeros((FLAGS.batch_size, self.max_input_length, FLAGS.word_emb_dim))
                    Y = np.zeros((FLAGS.batch_size, self.max_output_length, self.graph_vocab_codebook.size()))   

        # ignore left-over samples of incomplete batch


def train():
    parser = S2GParser()
    parser.setup(FLAGS.input_sent,FLAGS.input_graph)
    
    print 'Building model ...'
    #model = Sequential()
    model = SimpleSeq2seq(
        input_dim=FLAGS.word_emb_dim,
        input_length=parser.max_input_length,
        hidden_dim=FLAGS.hidden_dim,
        output_dim=parser.graph_vocab_codebook.size(),
        output_length=parser.max_output_length,
        depth=1
    )
    #model.add(seq2seq)
    model.compile(loss='categorical_crossentropy',optimizer='sgd')
    #model.compile(loss='mse',optimizer='rmsprop')

    print 'Begin training ...'
    start_time = time.time()
    total_iter = 1
    validation_frequency = parser.num_train_inst / FLAGS.batch_size # go over this number of batches and validate
    #for epoch in xrange(1, FLAGS.epoch+1):
    #    print 'epoch %d:' % (epoch)
    #    for X_train, Y_train in parser.get_training_batch(FLAGS.input_sent,FLAGS.input_graph):
    #        model.fit(X_train,Y_train, batch_size=FLAGS.batch_size, nb_epoch=1, show_accuracy=True, verbose=1)
    #        if total_iter % validation_frequency == 0:
    #            print 'Total iteration %d' % (total_iter)
    #        total_iter += 1
    n = 0
    n_train = parser.num_train_inst - parser.num_train_inst % FLAGS.batch_size
    X_train, Y_train = np.zeros((n_train, parser.max_input_length, FLAGS.word_emb_dim), dtype=np.float32), np.zeros((n_train, parser.max_output_length, parser.graph_vocab_codebook.size()),dtype=bool)
    
    for X, Y in parser.get_training_batch(FLAGS.input_sent, FLAGS.input_graph):
        #if X_train is None and Y_train is None:
        #    X_train = X
        #    Y_train = Y
        #else:
        if n % 10 == 0: print n
        X_train[n*FLAGS.batch_size:(n+1)*FLAGS.batch_size] = X
        Y_train[n*FLAGS.batch_size:(n+1)*FLAGS.batch_size] = Y
        n += 1

    #import pdb
    #pdb.set_trace()
    #X_train = np.asarray(X_train)
    #Y_train = np.asarray(Y_train)
            
    model.fit(X_train,Y_train, batch_size=FLAGS.batch_size, nb_epoch=FLAGS.epoch, show_accuracy=True, verbose=1)

    print 'Done training. %d epoches takes %.2f time total.' % (FLAGS.epoch, (time.time()-start_time) / 60.)
    
def test_batch():
    parser = S2GParser()
    parser.setup(FLAGS.input_sent,FLAGS.input_graph)


if __name__ == '__main__':
    gflags.DEFINE_string('input_sent','../train/token','input tokenized sentence')
    gflags.DEFINE_string('input_graph','../train/amrseq','input graph sequence')
    gflags.DEFINE_string('model_path','model/simple_seq2seq.m','model path')
    argv = FLAGS(sys.argv)
    #test_batch()
    train()


    
                
        

    