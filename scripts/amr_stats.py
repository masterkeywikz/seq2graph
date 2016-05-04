#!/usr/bin/python
import sys
import os
import argparse
from collections import defaultdict
class AMR_stats(object):
    def __init__(self, toks_=None, amr_seq_=None):
        self.num_reentrancy = 0
        self.num_predicates = defaultdict(int)
        self.num_nonpredicate_vals = defaultdict(int)
        self.num_consts = defaultdict(int)
        self.num_entities = defaultdict(int)
        self.num_relations = defaultdict(int)

        self.sentences = toks_
        self.amr_seqs = amr_seq_

    def update(self, local_re, local_pre, local_non, local_con, local_ent, local_rel):
        self.num_reentrancy += local_re
        for s in local_pre:
            self.num_predicates[s] += local_pre[s]

        for s in local_non:
            self.num_nonpredicate_vals[s] += local_non[s]

        for s in local_con:
            self.num_consts[s] += local_con[s]

        for s in local_ent:
            self.num_entities[s] += local_ent[s]

        for s in local_rel:
            self.num_relations[s] += local_rel[s]

    def dump2dir(self, dir):
        def dump_file(f, dict):
            sorted_dict = sorted(dict.items(), key=lambda k:(-k[1], k[0]))
            for (item, count) in sorted_dict:
                print >>f, '%s %d' % (item, count)
            f.close()

        pred_f = open(os.path.join(dir, 'pred'), 'w')
        non_pred_f = open(os.path.join(dir, 'non_pred_val'), 'w')
        const_f = open(os.path.join(dir, 'const'), 'w')
        entity_f = open(os.path.join(dir, 'entities'), 'w')
        relation_f = open(os.path.join(dir, 'relations'), 'w')

        dump_file(pred_f, self.num_predicates)
        dump_file(non_pred_f, self.num_nonpredicate_vals)
        dump_file(const_f, self.num_consts)
        dump_file(entity_f, self.num_entities)
        dump_file(relation_f, self.num_relations)

    def output_length_stats(self):
        if self.sentences:
            sentence_lengths = [len(toks) for toks in self.sentences]
            print 'Sentence length stats:'
            curr_num = 0
            for length in xrange(0, 101, 10):
                num_length_within = len([l for l in sentence_lengths if l >= length and l < length + 10])
                print 'length between %d and %d (exclude): %d' % (length, length+10, num_length_within)
            print 'length no smaller than 110: %d' % len([l for l in sentence_lengths if l >= 110])

        if self.amr_seqs:
            seq_lengths = [len(toks) for toks in self.amr_seqs]
            print 'AMR sequence length stats:'
            curr_num = 0
            for length in xrange(0, 301, 10):
                num_length_within = len([l for l in seq_lengths if l >= length and l < length + 10])
                print 'length between %d and %d (exclude): %d' % (length, length+10, num_length_within)
            print 'length no smaller than 310: %d' % len([l for l in seq_lengths if l >= 310])

    def prune_sentences(self, sent_length_thre, amrseq_length_thre, output_dir):
        assert self.sentences and self.amr_seqs
        remain_pairs = [(toks, amrseq) for (toks, amrseq) in zip(self.sentences, self.amr_seqs) if \
                len(toks) < sent_length_thre and len(amrseq) < amrseq_length_thre]
        output_sent_file = os.path.join(output_dir, ('toks_thred%d_%d' % (sent_length_thre, amrseq_length_thre)))
        output_amrseq_file = os.path.join(output_dir, ('amrseq_thred%d_%d' % (sent_length_thre, amrseq_length_thre)))

        with open(output_sent_file, 'w') as sent_f:
            with open(output_amrseq_file, 'w') as amrseq_f:
                for (toks, amrseq) in remain_pairs:
                    print >>sent_f, ' '.join(toks)
                    print >>amrseq_f, ' '.join(amrseq)

                amrseq_f.close()
                sent_f.close()

    def __str__(self):
        s = ''
        s += 'Total number of reentrancies: %d\n' % self.num_reentrancy
        s += 'Total number of predicates: %d\n' % len(self.num_predicates)
        s += 'Total number of non predicates variables: %d\n' % len(self.num_nonpredicate_vals)
        s += 'Total number of constants: %d\n' % len(self.num_consts)
        s += 'Total number of entities: %d\n' % len(self.num_entities)
        return s

def output_amr_stats(args):
    tok_file = os.path.join(args.data_dir, 'token')
    sents = [line.strip().split() for line in open(tok_file, 'r')]
    amr_seq_file = os.path.join(args.data_dir, 'amrseq')
    amr_seqs = [line.strip().split() for line in open(amr_seq_file, 'r')]

    stats = AMR_stats(sents, amr_seqs)
    stats.prune_sentences(args.sent_length, args.amrseq_length, args.output_dir)
    #stats.output_length_stats()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--data_dir", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--sent_length", type=int, help="the threshold for length of sentences")
    argparser.add_argument("--amrseq_length", type=int, help="the threshold for length of linearized amr sequences")
    argparser.add_argument("--output_dir", type=str, help="the output directory")

    args = argparser.parse_args()
    output_amr_stats(args)
