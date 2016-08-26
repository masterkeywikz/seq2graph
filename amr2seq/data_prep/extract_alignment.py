#!/usr/bin/python
import sys
import time
import os
import re
import argparse
from collections import defaultdict
from utils import read_toks, load_mappings
from entities import load_entities, identify_entities
from feature import *
def initialize_lemma(lemma_file):
    lemma_map = defaultdict(set)
    with open(lemma_file, 'r') as f:
        for line in f:
            fields = line.strip().split()
            word = fields[0]
            lemma = fields[1]
            if word == lemma:
                continue
            lemma_map[word].add(lemma)
    return lemma_map

def extract_features(args):

    tok_seqs = read_toks(args.tok_file)
    lemma_seqs = read_toks(args.lemma_file)
    pos_seqs = read_toks(args.pos_file)

    print 'A total of %d sentences' % len(tok_seqs)

    assert len(tok_seqs) == len(pos_seqs)
    assert len(tok_seqs) == len(lemma_seqs)

    stop_words = set([line.strip() for line in open(args.stop, 'r')])

    non_map_words = set(['am', 'is', 'are', 'be', 'a', 'an', 'the'])

    der_lemma_file = os.path.join(args.lemma_dir, 'der.lemma')
    der_lemma_map = initialize_lemma(der_lemma_file)

    train_entity_set = load_entities(args.stats_dir)
    all_entities = identify_entities(args.tok_file, args.ner_file, train_entity_set)

    (pred_set, pred_lemma_set, pred_labels, non_pred_mapping, non_pred_lemma_mapping, entity_labels) = load_mappings(args.stats_dir)

    feature_f = open(args.feature_file, 'w')
    #Each entity take the form (start, end, role)
    for (i, (toks, lemmas, pos_seq, entities_in_sent)) in enumerate(zip(tok_seqs, lemma_seqs, pos_seqs, all_entities)):
        n_toks = len(toks)
        aligned_set = set()

        all_spans = []
        assert len(toks) == len(lemmas)
        assert len(toks) == len(pos_seq)

        for (start, end, entity_typ) in entities_in_sent:
            new_aligned = set(xrange(start, end))
            aligned_set |= new_aligned
            all_spans.append((start, end, False, False, True))
            assert end <= len(toks)

        for index in xrange(n_toks):
            if index in aligned_set:
                continue

            curr_tok = toks[index]
            curr_lem = lemmas[index]
            curr_pos = pos_seq[index]

            aligned_set.add(index)

            if curr_tok in pred_set or curr_lem in pred_set or curr_lem in pred_lemma_set:
                all_spans.append((index, index+1, True, False, False))

            elif curr_tok in non_map_words:
                all_spans.append((index, index+1, False, False, False))
            elif curr_tok in non_pred_mapping or curr_lem in non_pred_lemma_mapping:
                all_spans.append((index, index+1, False, False, False))
            else: #not found in any mapping
                retrieved = False
                if curr_tok in der_lemma_map:
                    for tok in der_lemma_map[curr_tok]:
                        if tok in pred_set or tok in pred_lemma_set:
                            if curr_tok.endswith('ion') or curr_tok.endswith('er'):
                                all_spans.append((index, index+1, True, False, False))
                                retrieved = True
                                break
                if not retrieved:
                    all_spans.append((index, index+1, False, False, False))
            assert index < len(toks)

        all_spans = sorted(all_spans, key=lambda span: (span[0], span[1]))
        for (start, end, is_pred, is_op, is_ent) in all_spans:
            fs = []
            end -= 1
            #print start, end, len(toks)
            fs += extract_span(toks, start, end, 3, 'word')
            fs += extract_bigram(toks, start, end, 3, 'word')
            fs += extract_curr(toks, start, end, 'word')
            if is_ent:
                fs += extract_seq_feat(toks, start, end, 'word')

            #Lemma feature
            fs += extract_span(lemmas, start, end, 3, 'lemma')
            fs += extract_bigram(lemmas, start, end, 3, 'lemma')
            fs += extract_curr(lemmas, start, end, 'lemma')

            #Pos tag feature
            fs += extract_span(pos_seq, start, end, 3, 'POS')
            fs += extract_bigram(pos_seq, start, end, 3, 'POS')
            fs += extract_curr(pos_seq, start, end, 'POS')

            #Length of span feature
            fs.append('Length=%d' % (end - start))

            #Suffix feature
            if not is_ent and start == end:
                fs += suffix(toks[start])

            print >>feature_f, '##### %d-%d %s %s %s' % (start, end+1, '1' if is_pred else '0', '1' if is_ent else '0', ' '.join(fs))
        print >>feature_f, ''
    feature_f.close()




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--tok_file", type=str, help="the original tokenized text file", required=False)
    argparser.add_argument("--lemma_file", type=str, help="the lemmatized text file", required=False)
    argparser.add_argument("--pos_file", type=str, help="the pos tag file", required=False)
    argparser.add_argument("--ner_file", type=str, help="the NER file", required=False)
    argparser.add_argument("--feature_file", type=str, help="the result feature file", required=False)
    argparser.add_argument("--stats_dir", type=str, help="the statistics directory", required=False)
    argparser.add_argument("--output", type=str, help="the output file", required=False)
    argparser.add_argument("--lexical_grammar", type=str, help="the lexical grammar file", required=False)
    argparser.add_argument("--stop", type=str, help="stop words file", required=False)
    argparser.add_argument("--lemma_dir", type=str, help="lemma file directory", required=False)

    args = argparser.parse_args()
    extract_features(args)

