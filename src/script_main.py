#!/usr/bin/python
import ConfigParser
import time
import sys
import os
import numpy as np
from data_provider import *
import pdb
import theanets
import climate
import theano as T

logging = climate.get_logger(__name__)

climate.enable_default_logging()

def preProBuildWordVocab(sentence_iterator, word_count_threshold):
    # count up all word counts so that we can threshold
    # this shouldnt be too expensive of an operation
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    t0 = time.time()
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent['tokens']:
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d in %.2fs' % (len(word_counts), len(vocab), time.time() - t0)

    # with K distinct words:
    # - there are K+1 possible inputs (START token and all the words)
    # - there are K+1 possible outputs (END token and all the words)
    # we use ixtoword to take predicted indeces and map them to words for output visualization
    # we use wordtoix to take raw words and get their index in word vector matrix
    ixtoword = {}
    # period at the end of the sentence. make first dimension be end token
    ixtoword[0] = '.'
    wordtoix = {}
    wordtoix['#START#'] = 0  # make first vector be the start token
    wordtoix['#END#'] = 1 # make the 2nd vector be the end token
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    # compute bias vector, which is related to the log probability of the distribution
    # of the labels (words) and how often they occur. We will use this vector to initialize
    # the decoder weights, so that the loss function doesnt show a huge increase in performance
    # very quickly (which is just the network learning this anyway, for the most part). This makes
    # the visualizations of the cost function nicer because it doesn't look like a hockey stick.
    # for example on Flickr8K, doing this brings down initial perplexity from
    # ~2500 to ~170.
    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector)  # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)  # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

if __name__ == '__main__':

    '''
    # use more advanced approache to process the arguments. 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        help='dataset name (coco, flickr8k, flickr30k)')
    parser.add_argument('--h_size', default=256,
                        help='size of hidden layer')
    parser.add_argument('--e_size', default=256,
                        help='size of embed layer')
    parser.add_argument('--h_l1',default=0, help='hidden l1 reg')
    parser.add_argument('--h_l2',default=0, help='hidden l2 reg')
    parser.add_argument('--l1',default=0, help='l1 reg')
    parser.add_argument('--l2',default=0, help='l2 reg')
    parser.add_argument('--uid',default=None, help='uid for saving')
    
    args = parser.parse_args()
    
    if args.dataset is None or args.h_size is None or args.e_size is None:
        parser.print_help()
        sys.exit()
 
    dataset = args.dataset
    hid_size = args.h_size
    size = args.e_size
    uid = time.strftime('%d-%b-%Y-%H%M', time.gmtime())
    if args.uid:
        uid = args.uid

    h_l1 = float(args.h_l1)
    h_l2 = float(args.h_l2)
    l1 = float(args.l1)
    l2 = float(args.l2)
    '''

    cf = ConfigParser.ConfigParser()
    if len(sys.argv) < 2:
        print 'Usage: {0} <conf_fn> [n_words=3]'.format(sys.argv[0])
        sys.exit()
    n_words = 3
    if len(sys.argv) >= 3:
        n_words = int(sys.argv[2])
    cf.read(sys.argv[1])
    dataset = cf.get('INPUT', 'dataset')
    h_size=int(cf.get('INPUT','h_size'))
    e_size=int(cf.get('INPUT','e_size'))
    h_l1=float(cf.get('INPUT','h_l1'))
    h_l2=float(cf.get('INPUT','h_l2'))
    l1=float(cf.get('INPUT','l1'))
    l2=float(cf.get('INPUT','l2'))
    model_fn = cf.get('INPUT', 'model_fn')
    keywords_train_fn = cf.get('INPUT', 'keywords_train')
    keywords_val_fn = cf.get('INPUT', 'keywords_val')

    dropout = float(cf.get('INPUT', 'dropout'))

    save_dir=cf.get('OUTPUT', 'save_dir')
    #dataset = 'flickr8k'
    dp = getDataProvider(dataset)
    word_count_threshold = 5

    word2idx, idx2word, bias_init_vector = preProBuildWordVocab(
        dp.iterSentences('train'), word_count_threshold)

    # We also need to load the keywords first.
    dict_train_key_words = {}
    dict_val_key_words = {}
    with open(keywords_train_fn, 'r') as fid:
        for aline in fid:
            aline = aline.strip()
            words = aline.split()
            dict_train_key_words[ os.path.basename(words[0])] = words[1:]

    with open(keywords_val_fn, 'r') as fid:
        for aline in fid:
            aline = aline.strip()
            words = aline.split()
            dict_val_key_words[ os.path.basename(words[0])] = words[1:]

    batch_size = 256

    def vis_fea_len():
        batch = [dp.sampleImageSentencePair() for i in xrange(batch_size)]
        # May need to optimize here.
        # need to find out the longest length of a sentence.
        vis_fea_len = 0
        for i, pair in enumerate(batch):
            vis_fea_len = pair['image']['feat'].size
            return vis_fea_len

    def batch_train():
        batch = [dp.sampleImageSentencePair() for i in xrange(batch_size)]
        # May need to optimize here.
        # need to find out the longest length of a sentence.
        max_len = 0
        vis_fea_len = 0
        for i, pair in enumerate(batch):
            if max_len < len(pair['sentence']['tokens']):
                max_len = len(pair['sentence']['tokens'])
            vis_fea_len = pair['image']['feat'].size

        max_len += 2 # add both start and end token.
        fea = np.zeros(( max_len, batch_size, len(word2idx)), dtype='float32')
        label = np.zeros(( max_len - 1, batch_size), dtype='int32') # label do not need the #START# token.
        mask = np.zeros(( max_len - 1, batch_size), dtype='float32')

        vis_fea = np.zeros((batch_size, vis_fea_len), dtype='float32')
        #words_fea = np.asarray(np.random.rand(n_words, batch_size,len(word2idx)), dtype='float32')
        words_fea = np.zeros((n_words, batch_size,len(word2idx)), dtype='float32')

        for i, pair in enumerate(batch):
            img_fn = pair['image']['filename']
            key_words = dict_train_key_words[img_fn]

            vis_fea[i,:] = np.squeeze(pair['image']['feat'][:])
            tokens = ['#START#']
            tokens.extend(pair['sentence']['tokens'])
            tokens.append('#END#')
            #for j, w in enumerate(pair['sentence']['tokens']):
            for j, w in enumerate(tokens):
                if w in word2idx:
                    fea[ j, i, word2idx[w]] = 1.0
                    if j > 0:
                        mask[j-1, i] = 1.0
                        label[j-1,i] = word2idx[w]
            idx_word = 0
            for word in key_words:
                if word in word2idx:
                    words_fea[idx_word, i, word2idx[word]] = 1.0
                    idx_word += 1
                if idx_word >= n_words:
                    break
            if idx_word < n_words:
                logging.info('Not enough words provided for %s:%d', img_fn, idx_word)
        #logging.info('Max len for this batch %d',max_len)
        return [fea, label, mask, vis_fea, words_fea]

    def batch_val():
        batch = [dp.sampleImageSentencePair('val') for i in xrange(batch_size)]
        max_len = 0
        vis_fea_len = 0
        for i, pair in enumerate(batch):
            if max_len < len(pair['sentence']['tokens']):
                max_len = len(pair['sentence']['tokens'])
            vis_fea_len = pair['image']['feat'].size

        max_len += 2 # add both start and end token.

        fea = np.zeros(( max_len, batch_size, len(word2idx)), dtype='float32')
        label = np.zeros(( max_len-1, batch_size), dtype='int32')
        mask = np.zeros(( max_len-1, batch_size), dtype='float32')

        vis_fea = np.zeros((batch_size,vis_fea_len), dtype='float32')

        #words_fea = np.asarray(np.random.rand(n_words, batch_size, len(word2idx)), dtype='float32')
        words_fea = np.zeros((n_words, batch_size,len(word2idx)), dtype='float32')
        for i, pair in enumerate(batch):
            img_fn = pair['image']['filename']
            key_words = dict_val_key_words[img_fn]

            vis_fea[i,:] = np.squeeze(pair['image']['feat'][:])
            tokens = ['#START#']
            tokens.extend(pair['sentence']['tokens'])
            tokens.append('#END#')
            #for j, w in enumerate(pair['sentence']['tokens']):
            for j, w in enumerate(tokens):
                if w in word2idx:
                    fea[ j, i, word2idx[w]] = 1.0
                    if j > 0:
                        mask[j-1, i] = 1.0
                        label[j-1,i] = word2idx[w]

            idx_word = 0
            for word in key_words:
                if word in word2idx:
                    words_fea[idx_word, i, word2idx[word]] = 1.0
                    idx_word += 1
                if idx_word >= n_words:
                    break
            if idx_word < n_words:
                logging.info('Not enough words provided for %s:%d', img_fn, idx_word)

        return [fea, label, mask, vis_fea, words_fea]

    input_size = len(word2idx)
    vis_fea_len = vis_fea_len()
    def layer_lstm(n):
        return dict(form = 'lstm', size = n)
    def layer_lstm_emb(n,v):
        return dict(form = 'lstm', size = n, v_size = v)

    def layer_input_emb(n,v,i):
        return dict(size = n, v_size = v, input_size = i)

    def layer_input_emb_att(n,v,i,w):
        return dict(size = n, v_size = v, input_size = i, n_words = w)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    time_str = time.strftime("%d-%b-%Y-%H%M%S", time.gmtime())
    save_prefix = os.path.join(save_dir, os.path.splitext(os.path.basename(sys.argv[1]))[0] + '_' + str(dropout)+'_nwords_' + str(n_words))
    save_fn = save_prefix + '_' + time_str + '.pkl'
    logging.info('will save model to %s', save_fn)
    
    if os.path.isfile(model_fn):
        e = theanets.Experiment(model_fn)
    else:
        e = theanets.Experiment(
            theanets.recurrent.Classifier,
            layers=(layer_input_emb_att(e_size, vis_fea_len, input_size, n_words), layer_lstm(h_size),(input_size, 'softmax')),
            weighted=True,
            embedding=True,
            weak = True
        )
        e.train(
            batch_train,
            batch_val,
            algorithm='rmsprop',
            learning_rate=0.0001,
            momentum=0.9,
            max_gradient_clip=10,
            input_noise=0.0,
            train_batches=30,
            valid_batches=3,
            hidden_l1 = h_l1,
            hidden_l2 = h_l2,
            weight_l1 = l1,
            weight_l2 = l2,
            batch_size=batch_size,
            dropout=dropout,
            save_every = 100
        )

    e.train(
        batch_train,
        batch_val,
        algorithm='rmsprop',
        learning_rate=0.00001,
        momentum=0.9,
        max_gradient_clip=5,
        input_noise=0.0,
        train_batches=30,
        valid_batches=3,
        hidden_l1 = h_l1,
        hidden_l2 = h_l2,
        weight_l1 = l1,
        weight_l2 = l2,
        batch_size=batch_size,
        dropout=dropout,
        save_every = 100
    )
    e.save(save_fn)
