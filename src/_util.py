import numpy as np
unk_token = 'unk'
def load_vocab(vocab_fn):
    w2ix = {}
    ix2w = {}

    with open(vocab_fn,'r') as fid:
        for idx, aline in enumerate(fid):
            w = aline.strip()
            w2ix[w] = idx
            ix2w[idx] = w
    if unk_token not in w2ix:
        w2ix[unk_token] = len(w2ix)
        ix2w[len(ix2w)] = unk_token
    return w2ix, ix2w

start_tok="#START#"
end_tok="."
def load_vocab_dst(vocab_fn):
    # For dst, we will add the start and the end token.
    w2ix = {start_tok:0, end_tok:1}
    ix2w = {0:start_tok, 1:end_tok}
    # start_tok =  

    with open(vocab_fn,'r') as fid:
        idx = 2
        for aline in fid:
            w = aline.strip()
            if w not in w2ix:
                w2ix[w] = idx
                ix2w[idx] = w
                idx += 1
    if unk_token not in w2ix:
        w2ix[unk_token] = len(w2ix)
        ix2w[len(ix2w)] = unk_token
    return w2ix, ix2w

def load_split(w2ix, split_fn):
    max_seq_len = 0
    line_num = 0
    with open(split_fn, 'r') as fid:
        for aline in fid:
            toks = aline.strip().split()
            max_seq_len = max(max_seq_len, len(toks))
            line_num +=1

    np_split = np.zeros((line_num, max_seq_len), dtype = 'int32')
    np_split[:] = -1

    with open(split_fn,'r') as fid:
        for row, aline in enumerate(fid):
            toks = aline.strip().split()
            for col,tok in enumerate(toks):
                if tok in w2ix:
                    np_split[row, col] = w2ix[tok]
                else:
                    np_split[row, col] = w2ix[unk_token]

    return np_split
 
