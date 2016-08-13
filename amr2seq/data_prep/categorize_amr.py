#!/usr/bin/python
import sys
import time
import pickle
import os
import cPickle
from amr_graph import *
from amr_utils import *
import logger
import argparse
from re_utils import *
from preprocess import *
from collections import defaultdict
def build_bimap(tok2frags):
    frag2map = defaultdict(set)
    index2frags = defaultdict(set)
    for index in tok2frags:
        for frag in tok2frags[index]:
            index2frags[index].add(frag)
            frag2map[frag].add(index)

    return (index2frags, frag2map)

#Here we try to make the tok to fragment mapping one to one
def rebuild_fragment_map(tok2frags):
    (index2frags, frag2map) = build_bimap(tok2frags)
    for index in tok2frags:
        if len(tok2frags[index]) > 1:
            new_frag_list = []
            min_frag = None
            min_length = 100
            for frag in tok2frags[index]:
                index_set = frag2map[frag]
                assert index in index_set
                if len(index_set) > 1:
                    if len(index_set) < min_length:
                        min_length = len(index_set)
                        min_frag = frag
                    index_set.remove(index)
                else:
                    new_frag_list.append(frag)
            if len(new_frag_list) == 0:
                assert min_frag is not None
                new_frag_list.append(min_frag)
            tok2frags[index] = new_frag_list
    return tok2frags

def mergeSpans(index_to_spans):
    for index in index_to_spans:
        span_list = index_to_spans[index]
        new_span_list = []
        curr_start = None
        curr_end = None

        for (index, (start, end)) in enumerate(span_list):
            if curr_end is not None:
                assert start >= curr_end
                if start > curr_end: #There is a gap in between
                    new_span_list.append((curr_start, curr_end))
                    curr_start = start
                    curr_end = end
                else: #They equal, so update the end
                    curr_end = end

            else:
                curr_start = start
                curr_end = end

            if index + 1 == len(span_list): #Have reached the last position
                new_span_list.append((curr_start, curr_end))
        index_to_spans[index] = new_span_list
    return index_to_spans

def extractNodeMapping(alignments, amr_graph):
    node_to_span = defaultdict(list)
    edge_to_span = defaultdict(list)

    num_nodes = len(amr_graph.nodes)
    num_edges = len(amr_graph.edges)

    op_toks = []
    role_toks = []
    for curr_align in reversed(alignments):
        curr_tok = curr_align.split('-')[0]
        curr_frag = curr_align.split('-')[1]

        span_start = int(curr_tok)
        span_end = span_start + 1

        (index_type, index) = amr_graph.get_concept_relation(curr_frag)
        if index_type == 'c':
            node_to_span[index].append((span_start, span_end))
            curr_node = amr_graph.nodes[index]

            #Extract ops for entities
            if len(curr_node.p_edges) == 1:
                par_edge = amr_graph.edges[curr_node.p_edges[0]]
                if 'op' == par_edge.label[:2]:
                    op_toks.append((span_start, curr_node.c_edge))

            if curr_node.is_named_entity():
                role_toks.append((span_start, curr_node.c_edge))

        else:
            edge_to_span[index].append((span_start, span_end))
    node_to_span = mergeSpans(node_to_span)
    edge_to_span = mergeSpans(edge_to_span)

    return (op_toks, role_toks, node_to_span, edge_to_span)

def extract_fragments(alignments, amr_graph):
    tok2frags = defaultdict(list)

    num_nodes = len(amr_graph.nodes)
    num_edges = len(amr_graph.edges)

    op_toks = []
    role_toks = []
    for curr_align in reversed(alignments):
        curr_tok = curr_align.split('-')[0]
        curr_frag = curr_align.split('-')[1]

        span_start = int(curr_tok)
        span_end = span_start + 1

        (index_type, index) = amr_graph.get_concept_relation(curr_frag)
        frag = AMRFragment(num_edges, num_nodes, amr_graph)
        if index_type == 'c':
            frag.set_root(index)
            curr_node = amr_graph.nodes[index]

            #Extract ops for entities
            if len(curr_node.p_edges) == 1:
                par_edge = amr_graph.edges[curr_node.p_edges[0]]
                if 'op' == par_edge.label[:2]:
                    op_toks.append((span_start, curr_node.c_edge))

            if curr_node.is_named_entity():
                role_toks.append((span_start, curr_node.c_edge))

            frag.set_edge(curr_node.c_edge)

        else:
            frag.set_edge(index)
            curr_edge = amr_graph.edges[index]
            frag.set_root(curr_edge.head)
            frag.set_node(curr_edge.tail)

        frag.build_ext_list()
        frag.build_ext_set()

        tok2frags[span_start].append(frag)

    for index in tok2frags:
        if len(tok2frags[index]) > 1:
            tok2frags[index] = connect_adjacent(tok2frags[index], logger)

    tok2frags = rebuild_fragment_map(tok2frags)
    for index in tok2frags:
        for frag in tok2frags[index]:
            frag.set_span(index, index+1)

    return (op_toks, role_toks, tok2frags)

#Verify this fragment contains only one edge and return it
def unique_edge(frag):
    #assert frag.edges.count() == 1, 'Not unify edge fragment found'
    amr_graph = frag.graph
    edge_list = []
    n_edges = len(frag.edges)
    for i in xrange(n_edges):
        if frag.edges[i] == 1:
            edge_list.append(i)
    assert len(edge_list) == frag.edges.count()
    return tuple(edge_list)

class AMR_stats(object):
    def __init__(self):
        self.num_reentrancy = 0
        self.num_predicates = defaultdict(int)
        self.num_nonpredicate_vals = defaultdict(int)
        self.num_consts = defaultdict(int)
        self.num_named_entities = defaultdict(int)
        self.num_entities = defaultdict(int)
        self.num_relations = defaultdict(int)

    def update(self, local_re, local_pre, local_non, local_con, local_ent, local_ne):
        self.num_reentrancy += local_re
        for s in local_pre:
            self.num_predicates[s] += local_pre[s]

        for s in local_non:
            self.num_nonpredicate_vals[s] += local_non[s]

        for s in local_con:
            self.num_consts[s] += local_con[s]

        for s in local_ent:
            self.num_entities[s] += local_ent[s]

        for s in local_ne:
            self.num_named_entities[s] += local_ne[s]
        #for s in local_rel:
        #    self.num_relations[s] += local_rel[s]

    def collect_stats(self, amr_graphs):
        for amr in amr_graphs:
            (named_entity_nums, entity_nums, predicate_nums, variable_nums, const_nums, reentrancy_nums) = amr.statistics()
            self.update(reentrancy_nums, predicate_nums, variable_nums, const_nums, entity_nums, named_entity_nums)

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
        named_entity_f = open(os.path.join(dir, 'named_entities'), 'w')
        #relation_f = open(os.path.join(dir, 'relations'), 'w')

        dump_file(pred_f, self.num_predicates)
        dump_file(non_pred_f, self.num_nonpredicate_vals)
        dump_file(const_f, self.num_consts)
        dump_file(entity_f, self.num_entities)
        dump_file(named_entity_f, self.num_named_entities)
        #dump_file(relation_f, self.num_relations)

    def loadFromDir(self, dir):
        def load_file(f, dict):
            for line in f:
                item = line.strip().split(' ')[0]
                count = int(line.strip().split(' ')[1])
                dict[item] = count
            f.close()

        pred_f = open(os.path.join(dir, 'pred'), 'r')
        non_pred_f = open(os.path.join(dir, 'non_pred_val'), 'r')
        const_f = open(os.path.join(dir, 'const'), 'r')
        entity_f = open(os.path.join(dir, 'entities'), 'r')
        named_entity_f = open(os.path.join(dir, 'named_entities'), 'r')

        load_file(pred_f, self.num_predicates)
        load_file(non_pred_f, self.num_nonpredicate_vals)
        load_file(const_f, self.num_consts)
        load_file(entity_f, self.num_entities)
        load_file(named_entity_f, self.num_named_entities)

    def __str__(self):
        s = ''
        s += 'Total number of reentrancies: %d\n' % self.num_reentrancy
        s += 'Total number of predicates: %d\n' % len(self.num_predicates)
        s += 'Total number of non predicates variables: %d\n' % len(self.num_nonpredicate_vals)
        s += 'Total number of constants: %d\n' % len(self.num_consts)
        s += 'Total number of entities: %d\n' % len(self.num_entities)
        s += 'Total number of named entities: %d\n' % len(self.num_named_entities)

        return s

def linearize_amr(args):
    logger.file = open(os.path.join(args.run_dir, 'logger'), 'w')

    amr_file = os.path.join(args.data_dir, 'amr')
    alignment_file = os.path.join(args.data_dir, 'alignment')
    if args.use_lemma:
        tok_file = os.path.join(args.data_dir, 'lemmatized_token')
    else:
        tok_file = os.path.join(args.data_dir, 'token')
    pos_file = os.path.join(args.data_dir, 'pos')

    collapsed_tok_file = os.path.join(args.data_dir, 'collapsed_token')
    collapsed_graph_file = os.path.join(args.data_dir, 'collased_amr')
    map_file = os.path.join(args.data_dir, 'map')

    amr_graphs = load_amr_graphs(amr_file)
    alignments = [line.strip().split() for line in open(alignment_file, 'r')]
    toks = [line.strip().split() for line in open(tok_file, 'r')]
    poss = [line.strip().split() for line in open(pos_file, 'r')]

    assert len(amr_graphs) == len(alignments) and len(amr_graphs) == len(toks) and len(amr_graphs) == len(poss), '%d %d %d %d %d' % (len(amr_graphs), len(alignments), len(toks), len(poss))

    num_self_cycle = 0
    used_sents = 0

    amr_statistics = AMR_stats()

    if args.use_stats:
        amr_statistics.loadFromDir(args.stats_dir)
        print amr_statistics
    else:
        os.system('mkdir -p %s' % args.stats_dir)
        amr_statistics.collect_stats(amr_graphs)
        amr_statistics.dump2dir(args.stats_dir)

    singleton_num = 0.0
    multiple_num = 0.0
    total_num = 0.0
    empty_num = 0.0

    for (sent_index, (tok_seq, pos_seq, alignment_seq, amr)) in enumerate(zip(toks, poss, alignments, amr_graphs)):

        logger.writeln('Sentence #%d' % (sent_index+1))
        logger.writeln(' '.join(tok_seq))

        edge_alignment = bitarray(len(amr.edges))
        if edge_alignment.count() != 0:
            edge_alignment ^= edge_alignment
        assert edge_alignment.count() == 0

        has_cycle = False
        if amr.check_self_cycle():
            num_self_cycle += 1
            has_cycle = True

        amr.set_sentence(tok_seq)
        amr.set_poss(pos_seq)

        aligned_fragments = []
        reentrancies = {}  #Map multiple spans as reentrancies, keeping only one as original, others as connections

        has_multiple = False
        no_alignment = False

        aligned_set = set()

        (op_toks, role_toks, node_to_span, edge_to_span) = extractNodeMapping(alignment_seq, amr)

        #if not aligned_fragments:
        #    logger.writeln('no alignments')
        #    continue

        #temp_aligned = set(aligned_fragments.keys())
        aligned_fragments = sorted(aligned_fragments.items(), key=lambda frag: frag[0])

        temp_unaligned = set(xrange(len(pos_seq))) - temp_aligned

        all_frags = []
        all_alignments = defaultdict(list)

        ####Extract entities mapping#####
        for (frag, frag_label) in amr.extract_entities():
            if len(opt_toks) == 0:
                logger.writeln("No alignment for the entity found")

            (aligned_indexes, entity_spans) = all_aligned_spans(frag, opt_toks, role_toks, temp_unaligned)
            root_node = amr.nodes[frag.root]

            entity_mention_toks = root_node.namedEntityMention()
            print 'fragment entity mention: %s' % ' '.join(entity_mention_toks)

            total_num += 1.0
            if entity_spans:
                entity_spans = removeRedundant(tok_seq, entity_spans, entity_mention_toks)
                if len(entity_spans) == 1:
                    singleton_num += 1.0
                    logger.writeln('Single fragment')
                    for (frag_start, frag_end) in entity_spans:
                        logger.writeln(' '.join(tok_seq[frag_start:frag_end]))
                        all_alignments[frag.root].append((frag_start, frag_end))
                else:
                    multiple_num += 1.0
                    logger.writeln('Multiple fragment')
                    logger.writeln(aligned_indexes)
                    logger.writeln(' '.join([tok_seq[index] for index in aligned_indexes]))

                    for (frag_start, frag_end) in entity_spans:
                        logger.writeln(' '.join(tok_seq[frag_start:frag_end]))
                        all_alignments[frag.root].append((frag_start, frag_end))
            else:
                empty_num += 1.0
                _ = all_alignments[frag.root]

            #new_aligned = set(xrange(frag_start, frag_end))
            #if len(new_aligned & aligned_set) != 0:
            #    print str(amr)
            #    print str(frag)
            #    has_multiple = True
            #    break
            #    #continue

            #aligned_set |= new_aligned
            #all_frags.append(frag)

            #if (edge_alignment & frag.edges).count() != 0:
            #    has_multiple = True

            #edge_alignment |= frag.edges

        #if no_alignment:
        #    continue

        #one2many = False
        #######Extra other alignments######
        #logger.writeln('Aligned fragments:')
        #for (index, frag_list) in aligned_fragments:
        #    if index in aligned_set:
        #        continue

        #    assert len(frag_list) > 0
        #    non_conflict = 0
        #    non_conflict_list = []
        #    for frag in frag_list:
        #        if (edge_alignment & frag.edges).count() == 0:
        #            non_conflict += 1
        #            non_conflict_list.append(frag)

        #    if non_conflict != 1:
        #        one2many = True

        #    used_frag = None
        #    if non_conflict == 0:
        #        used_frag = frag_list[0]
        #    else:
        #        used_frag = non_conflict_list[0]

        #    edge_alignment |= used_frag.edges
        #    all_frags.append(used_frag)

        #    aligned_set.add(index)

        #logger.writeln("%d aligned edges out of %d total" % (edge_alignment.count(), len(edge_alignment)))
        #used_sents += 1

        assert len(tok_seq) == len(pos_seq)

        #unaligned_toks = [(i, tok) for (i, tok) in enumerate(tok_seq) if i not in aligned_set]
        #(aligned, unaligned) = amr.recall_unaligned_concepts(edge_alignment, unaligned_toks, lemma_map, stop_words)
        #aligned = [x for (x, y, z, k) in aligned]

        #all_frags += aligned
    print "one to one alignment: %lf" % (singleton_num/total_num)
    print "one to multiple alignment: %lf" % (multiple_num/total_num)
    print "one to empty alignment: %lf" % (empty_num/total_num)
    #amr_statistics.dump2dir(args.run_dir)
    #print str(amr_statistics)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--amr_file", type=str, help="the original AMR graph files", required=False)
    argparser.add_argument("--stop", type=str, help="stop words file", required=False)
    argparser.add_argument("--lemma", type=str, help="lemma file", required=False)
    argparser.add_argument("--data_dir", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--run_dir", type=str, help="the output directory for saving the constructed forest")
    argparser.add_argument("--use_lemma", action="store_true", help="if use lemmatized tokens")
    argparser.add_argument("--use_stats", action="store_true", help="if use a built-up statistics")
    argparser.add_argument("--stats_dir", type=str, help="the statistics directory")
    argparser.add_argument("--min_prd_freq", type=int, default=50, help="threshold for filtering predicates")
    argparser.add_argument("--min_var_freq", type=int, default=50, help="threshold for filtering non predicate variables")
    argparser.add_argument("--index_unknown", action="store_true", help="if to index the unknown predicates or non predicate variables")

    args = argparser.parse_args()
    linearize_amr(args)
