import sys
import time
import os

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print "Usage: {0} <word_vec_fn> <vocab_fn>".format(sys.argv[0])
        sys.exit()
    
    word_vec_fn = sys.argv[1]
    vocab_fn = sys.argv[2]

    save_fn = os.path.splitext(vocab_fn)[0] + '_glove.txt'
    fea_save_fn = os.path.splitext(vocab_fn)[0] + '_fea.txt'

    vocab_dict = {}
    with open(vocab_fn) as fid:
        for aline in fid:
            w = aline.strip()
            vocab_dict[w] = 1
    # add end token.
    vocab_dict['.'] = 1

    word2vec_dict = {}

    with open(word_vec_fn,'r') as fid:
        for aline in fid:
            aline = aline.strip()
            parts = aline.split()
            if parts[0] in vocab_dict:
                word2vec_dict[parts[0]] = 1
    idx2word = {}
    # period at the end of the sentence. make first dimension be end token
    word2idx = {}
    idx = 0 
    for w in word2vec_dict:
        if w not in word2idx:
            word2idx[w] = idx
            idx2word[idx] = w
            idx += 1

    with open(save_fn, 'w') as fid:
        for i in range(len(idx2word)):
            print >>fid, idx2word[i]
    with open(fea_save_fn,'w') as wfid:
        with open(word_vec_fn,'r') as fid:
            for aline in fid:
                aline = aline.strip()
                parts = aline.split()
                if parts[0] in word2idx:
                    print>>wfid, aline
    print 'Done with vocab', save_fn, fea_save_fn
