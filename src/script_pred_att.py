#!/usr/bin/python

import theanets
import numpy as np
import scipy.io
import sys
import pdb
import os
import climate
import time
from _util import * 
logging = climate.get_logger(__name__)
climate.enable_default_logging()

if __name__ == '__main__':

    if len(sys.argv) < 5:
        print 'Usage: {0} <saved_model> <test_src_vocab> <test_src_fn> <test_dst_vocab> [beam_size=10] [batch_size=32]'.format(sys.argv[0])
        sys.exit()

    model_fn = sys.argv[1]
    src_vocab_fn = sys.argv[2]
    src_fn = sys.argv[3]
    dst_vocab_fn = sys.argv[4]
    beam_size = 10
    batch_size = 32
    dropout = 0.0
   
    if len(sys.argv) >=6:
        beam_size = int(sys.argv[5])

    if len(sys.argv) >= 7:
        batch_size = int(sys.argv[6])

    src_w2ix, src_ix2w = load_vocab(src_vocab_fn)
    dst_w2ix, dst_ix2w = load_vocab_dst(dst_vocab_fn)
    src_tst_np = load_split(src_w2ix, src_fn)

    src = np.zeros((src_tst_np.shape[1], batch_size, len(src_w2ix)), dtype = 'float32')
    src_mask = np.zeros((src_tst_np.shape[1], batch_size), dtype = 'float32')
    logging.info("Loading net work from %s", model_fn)
    e = theanets.Experiment(model_fn)
    network = e.network
    logging.info('Done with loading model %s', model_fn)
    all_candidates = []

    save_dir = os.path.splitext(os.path.basename(model_fn))[0] + '_' + str(beam_size)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pred_fn = os.path.join(save_dir, 'pred_fn.txt')
    loss_fn = os.path.join(save_dir, 'pred_loss.txt')
    dst_fid = open(pred_fn,'w')
    dst_loss_fid = open(loss_fn,'w')
    kwargs = {}
    for i in range(0,src_tst_np.shape[0], batch_size):
        src[:] = 0
        src_mask[:] = 0
        start_idx = i
        end_idx = min(i + batch_size, src_tst_np.shape[0])

        for i in range(start_idx, end_idx):
            src_i = src_tst_np[i,:]

            for j,pos in enumerate(src_i):
                if pos < 0:
                    break
                src[j, i-start_idx, pos] = 1
                src_mask[j, i-start_idx] = 1
        src = src[:,0:(end_idx - start_idx),:]
        src_mask = src_mask[:,0:(end_idx - start_idx)]

        logging.info('Predicting %d~%d/%d', start_idx, end_idx, src_tst_np.shape[0])
        beams = network.predict_captions_forward_batch(src, src_mask, beam_size, **kwargs)
        for beam in beams:
            top_prediction = beam[0]
            # ix 0 is the END token, skip that
            candidate = ' '.join([dst_ix2w[ix] for ix in top_prediction[1] if ix > 0])
            print >>dst_fid, candidate
            print >>dst_loss_fid, top_prediction[0] 
    
    logging.info('Done with %s', pred_fn) 
    logging.info('Done with %s', loss_fn) 
