import theano.tensor as TT
import theano
#import util
import numpy as np
import pdb

#batch_size = 256
#leng = 10
#fea = 200
#v_fea = 1025
#nout = 512
#
#n_words = 4
#
#wt = theano.shared(util.random_matrix(fea, nout),name='wt')
#bt = theano.shared(util.random_vector(nout),name='bt')
#
#we = theano.shared(util.random_matrix(v_fea, nout),name='we')
#be = theano.shared(util.random_vector( nout),name='be')
#
#
#wt_w = theano.shared(util.random_matrix(fea, nout), name= 'wt_w')
#bt_w = theano.shared(util.random_vector(nout), name='bt_w')
#
#v_a = theano.shared(util.random_vector(nout), name='v_a')
#b_a = theano.shared(util.random_vector(n_words), name='b_a')
#v_t = theano.shared(util.random_matrix(nout, nout), name='v_t')
#b_vt = theano.shared(util.random_vector(nout), name = 'b_vt')
#
#
#x = TT.ftensor3('x')
v = TT.matrix('v')
def fun(x):
    return TT.nonzero(x, True)

def step(x):
    return TT.max(TT.nonzero(x,True))

def test(x3, idx, idx2):
    return x3[idx, idx2,:]

def test2(idx_row, idxcol, x3):
    return x3[idx_row, idxcol,:]

def test3(idxs, x3):
    return x3[idxs[0,:], idxs[1,:], :]

def test4(x3):
    return TT.nonzero_values(x3)

v3 = TT.ftensor3('v')
idx = TT.imatrix('i')
idx2 = TT.imatrix('i2')

output_test2 = test2(idx, idx2, v3)
output_test3 = test3(idx, v3)
output4 = test4(v3)

x_scan, updates = theano.scan(test2, 
        outputs_info=None, 
        sequences=[idx, idx2], 
        non_sequences = [v3]
)

func = theano.function(inputs = [idx,idx2, v3], outputs = x_scan)
func2 = theano.function(inputs = [idx, idx2, v3], outputs = output_test2)
#func3 = theano.function(inputs = [idx, v3], outputs = output_test3)
func4 = theano.function(inputs = [v3], outputs = output4)

x3 = np.random.rand(3,4,2).astype('float32')
idx = np.array([[0,1,2]],dtype = 'int32')
idx2 = np.array([[2,2,1]], dtype = 'int32')

x3[:] = 0
x3[1,1,:] = 1.0
x3[2,2,:] = 2
x3[0,1,:] = 3

print x3
print x3.shape
print func4(x3)




#words = TT.ftensor3('words')
#
#dim_words = words.dimshuffle((1,0,2)) # now x is batch * n_words * dict_len
#dim_x = x.dimshuffle((1,0,2))
#
#def step_(x_t, words_t, w_t, b_t, w_t_w, b_t_w, v_t, b_vt, v_a, b_a):
#    TT_x = TT.dot(x_t, w_t) + b_t
#    TT_w = TT.dot(words_t, w_t_w) + b_t_w # 10 * 512
#    TT_w_dim = TT_w.dimshuffle(('x', 0, 1))
#    TT_ws = TT.extra_ops.repeat(TT_w_dim, TT_x.shape[0], axis = 0) # sentenlen * n_words * 512
#
#    TT_xv = TT.dot(TT_x, v_t) + b_vt # sentence_len * 512
#    TT_xv_dim = TT_xv.dimshuffle(0,'x', 1)
#    TT_xvs = TT.extra_ops.repeat(TT_xv_dim, TT_ws.shape[1], axis = 1)
#    TT_act = TT.tanh( TT_ws + TT_xvs ) # sentence * n_words * 512
#    beta = TT.dot(TT_act, v_a ) + b_a # this is broadcastable: sentence * n_words
#    z = TT.exp(beta - beta.max(axis=-1, keepdims=True))
#    alpha = z / z.sum(axis=-1, keepdims=True) # sentence * n_words.
#    TT_att = TT.dot(alpha, TT_w) # now, good, sentence * 512
#    return TT.concatenate((TT_x, TT_att), axis = 1)
#    #return TT.concatenate((TT_x, TT.extra_ops.repeat(TT_att_f, TT_x.shape[0], axis = 0)), axis = 1)
#
#x_scan, updates = theano.scan(step_, 
#        outputs_info=None, 
#        sequences=[dim_x, dim_words], 
#        non_sequences=[wt, 
#            bt, 
#            wt_w, 
#            bt_w, 
#            v_t, 
#            b_vt,
#            v_a,
#            b_a
#            ]
#        )
#
#func = theano.function(inputs = [ x, words], outputs = [x_scan], updates = updates)
##x_scan_simple, updates_simple = theano.scan(step_simple, outputs_info=None, sequences=[x], non_sequences=[wt, bt] )
##x_e = x_scan.dimshuffle((1,0,2))
##x_e_simple = x_scan_simple.dimshuffle((1,0,2))
###x_e_simple = x_scan_simple
##func = theano.function(inputs = [ x, words], outputs = [x_e], updates = updates)
##func_simple = theano.function(inputs = [ x], outputs = [x_e_simple], updates = updates)
##
##def append_(x_t, word_t, w_t, b_t):
##            TT_w = TT.dot(word_t, w_t) + b_t
##            return TT.concatenate((x_t, TT.flatten(TT_w)))
##
##x_scan_0, updates_app = theano.scan(append_, outputs_info = None, sequences = [x_0, words], non_sequences = [ self.find('wt'), self.find('bt')])
##x_0 = x_scan_0.dimshuffle(('x',0,1))
#
#
#np_x = np.asarray(np.random.rand(leng, batch_size, fea), dtype='float32')
#np_v = np.asarray(np.random.rand(batch_size, fea), dtype='float32')
#np_words = np.asarray(np.random.rand(n_words,batch_size, fea), dtype='float32')
#
##np_x = np.asarray(np.random.rand(batch_size, leng, fea), dtype='float32')
##np_words = np.asarray(np.random.rand(batch_size, 4, fea), dtype='float32')
#print np_x.shape
#print np_words.shape
#test = func(np_x, np_words)
##test_simple = func_simple(np_x)
#print 'test',test[0].shape
#
#print test[0][0,:]
#print test[0][0,:].shape
#print test[0][0,:].sum(axis = 0)
#print test[0][0,:].sum(axis = 1)
#
#
##
### Now, we start the visual word.
##x_0 = TT.dot(v, self.find('we')) + self.find('be') # now batch_size * e_size
##def append_(x_t, word_t, w_t, b_t):
##    TT_w = TT.dot(word_t, w_t) + b_t
##    return TT.concatenate((x_t, TT.flatten(TT_w)))
##
##x_scan_0, updates_app = theano.scan(append_, outputs_info = None, sequences = [x_0, dim_words], non_sequences = [ we, be])
##
##x_0 = x_scan_0.dimshuffle(('x',0,1))
###
##y = TT.concatenate((x_0, x_e))
##func_x0 = theano.function(inputs = [ v, words ], outputs = [x_0], updates = updates_app)
##
##test_x0 = func_x0(np_v, np_words)
#
##print 'test_x0', test_x0[0].shape
