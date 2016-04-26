#!/usr/bin/python

import sys
import pdb
import os
import climate
import time
import nltk.data 
import nltk.tokenize
import string

logging = climate.get_logger(__name__)
climate.enable_default_logging()

if __name__ == '__main__':

    if len(sys.argv) < 2:
        logging.info('Usage: {0} <fn> [threshod=5]'.format(sys.argv[0]))
        sys.exit()
    
    fn_txt = sys.argv[1]
    word_count_threshold = 5

    if len(sys.argv) >= 3:
        word_count_threshold = int(sys.argv[2])

    save_fn = os.path.splitext(fn_txt)[0] + '_' + str(word_count_threshold) + '_vocab.txt'
    dict_tok = {}
    
    with open(fn_txt) as fid:
        for aline in fid:
            toks = aline.strip().split()
            for tok in toks:
                if tok not in dict_tok:
                    dict_tok[tok] = 0
                dict_tok[tok] += 1
    
    toks = [ tok for tok in dict_tok if  dict_tok[tok] >= word_count_threshold]
    with open(save_fn,'w') as fid:
        for w in toks:
            print>>fid, w
    print 'Done with', save_fn
