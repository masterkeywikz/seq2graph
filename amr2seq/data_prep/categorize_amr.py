#!/usr/bin/python
import sys
import time
import pickle
import os
import re
import cPickle
from amr_graph import *
from amr_utils import *
import logger
import argparse
from re_utils import *
from preprocess import *
from collections import defaultdict
from entities import identify_entities

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
    new_index_to_spans = {}
    for index in index_to_spans:
        span_list = index_to_spans[index]
        span_list = sorted(span_list, key=lambda x:x[1])
        new_span_list = []
        curr_start = None
        curr_end = None

        for (idx, (start, end, _)) in enumerate(span_list):
            if curr_end is not None:
                #assert start >= curr_end, span_list
                if start < curr_end:
                    continue
                if start > curr_end: #There is a gap in between
                    new_span_list.append((curr_start, curr_end, None))
                    curr_start = start
                    curr_end = end
                else: #They equal, so update the end
                    curr_end = end

            else:
                curr_start = start
                curr_end = end

            if idx + 1 == len(span_list): #Have reached the last position
                new_span_list.append((curr_start, curr_end, None))
        new_index_to_spans[index] = new_span_list
    return new_index_to_spans

def extractNodeMapping(alignments, amr_graph):
    aligned_set = set()

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

        aligned_set.add(span_start)

        (index_type, index) = amr_graph.get_concept_relation(curr_frag)
        if index_type == 'c':
            node_to_span[index].append((span_start, span_end, None))
            curr_node = amr_graph.nodes[index]

            #Extract ops for entities
            if len(curr_node.p_edges) == 1:
                par_edge = amr_graph.edges[curr_node.p_edges[0]]
                if 'op' == par_edge.label[:2]:
                    op_toks.append((span_start, curr_node.c_edge))

            if curr_node.is_named_entity():
                role_toks.append((span_start, curr_node.c_edge))

        else:
            edge_to_span[index].append((span_start, span_end, None))
    new_node_to_span = mergeSpans(node_to_span)
    new_edge_to_span = mergeSpans(edge_to_span)

    return (op_toks, role_toks, new_node_to_span, new_edge_to_span, aligned_set)

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

#Traverse AMR from top down, also categorize the sequence in case of alignment existed
def categorizeParallelSequences(amr, tok_seq, all_alignments, pred_freq_thre=50, var_freq_thre=50):

    old_depth = -1
    depth = -1
    stack = [(amr.root, TOP, None, 0)] #Start from the root of the AMR
    aux_stack = []
    seq = []

    cate_tok_seq = []
    ret_index = 0

    seq_map = {}   #Map each span to a category
    visited = set()

    covered = set()
    multiple_covered = set()

    node_to_label = {}
    cate_to_index = {}

    while stack:
        old_depth = depth
        curr_node_index, rel, parent, depth = stack.pop()
        curr_node = amr.nodes[curr_node_index]
        curr_var = curr_node.node_str()

        i = 0

        while old_depth - depth >= i:
            if aux_stack == []:
                import pdb
                pdb.set_trace()
            seq.append(aux_stack.pop())
            i+=1

        if curr_node_index in visited: #A reentrancy found
            seq.append((rel+LBR, None))
            seq.append((RET + ('-%d' % ret_index), None))
            ret_index += 1
            aux_stack.append((RBR+rel, None))
            continue

        visited.add(curr_node_index)
        seq.append((rel+LBR, None))

        exclude_rels, cur_symbol, categorized = amr.get_symbol(curr_node_index, pred_freq_thre, var_freq_thre)

        if categorized:
            node_to_label[curr_node_index] = cur_symbol

        seq.append((cur_symbol, curr_node_index))
        aux_stack.append((RBR+rel, None))

        for edge_index in reversed(curr_node.v_edges):
            curr_edge = amr.edges[edge_index]
            child_index = curr_edge.tail
            if curr_edge.label in exclude_rels:
                continue
            stack.append((child_index, curr_edge.label, curr_var, depth+1))

    seq.extend(aux_stack[::-1])
    cate_span_map, end_index_map, covered_toks = categorizedSpans(all_alignments, node_to_label)

    map_seq = []
    nodeindex_to_tokindex = {}  #The mapping
    label_to_index = defaultdict(int)

    for tok_index, tok in enumerate(tok_seq):
        if tok_index not in covered_toks:
            cate_tok_seq.append(tok)
            align_str = '%d-%d++%s++NONE++NONE++NONE' % (tok_index, tok_index+1, tok)
            map_seq.append(align_str)
            continue

        if tok_index in end_index_map: #This span can be mapped to category
            end_index = end_index_map[tok_index]
            assert (tok_index, end_index) in cate_span_map

            node_index, aligned_label, wiki_label = cate_span_map[(tok_index, end_index)]

            if node_index not in nodeindex_to_tokindex:
                nodeindex_to_tokindex[node_index] = defaultdict(int)

            indexed_aligned_label = '%s-%d' % (aligned_label, label_to_index[aligned_label])
            nodeindex_to_tokindex[node_index] = indexed_aligned_label
            label_to_index[aligned_label] += 1


            cate_tok_seq.append(indexed_aligned_label)

            #align_str = '%d-%d:%s:%d:%s:%s' % (tok_index, end_index, ' '.join(tok_seq[tok_index:end_index]), node_index, amr.nodes[node_index].node_str(), indexed_aligned_label)

            align_str = '%d-%d++%s++%s++%s++%s' % (tok_index, end_index, ' '.join(tok_seq[tok_index:end_index]), wiki_label if wiki_label is not None else 'NONE', amr.nodes[node_index].node_str(), aligned_label)
            map_seq.append(align_str)

    seq = [nodeindex_to_tokindex[node_index] if node_index in nodeindex_to_tokindex else label for (label, node_index) in seq]

    return seq, cate_tok_seq, map_seq

def categorizedSpans(all_alignments, node_to_label):
    visited = set()

    all_alignments = sorted(all_alignments.items(), key=lambda x:len(x[1]))
    span_map = {}
    end_index_map = {}

    for (node_index, aligned_spans) in all_alignments:
        if node_index in node_to_label:
            aligned_label = node_to_label[node_index]
            for (span_start, span_end, wiki_label) in aligned_spans:
                span_set = set(xrange(span_start, span_end))
                if len(span_set & visited) != 0:
                    continue

                visited |= span_set

                span_map[(span_start, span_end)] = (node_index, aligned_label, wiki_label)
                end_index_map[span_start] = span_end

    return span_map, end_index_map, visited

def linearize_amr(args):
    logger.file = open(os.path.join(args.run_dir, 'logger'), 'w')

    amr_file = os.path.join(args.data_dir, 'amr')
    alignment_file = os.path.join(args.data_dir, 'alignment')
    if args.use_lemma:
        tok_file = os.path.join(args.data_dir, 'lemmatized_token')
    else:
        tok_file = os.path.join(args.data_dir, 'token')
    pos_file = os.path.join(args.data_dir, 'pos')

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

    if args.parallel:
        singleton_num = 0.0
        multiple_num = 0.0
        total_num = 0.0
        empty_num = 0.0

        amr_seq_file = os.path.join(args.run_dir, 'amrseq')
        tok_seq_file = os.path.join(args.run_dir, 'tokseq')
        map_seq_file = os.path.join(args.run_dir, 'train_map')

        amrseq_wf = open(amr_seq_file, 'w')
        tokseq_wf = open(tok_seq_file, 'w')
        mapseq_wf = open(map_seq_file, 'w')

        for (sent_index, (tok_seq, pos_seq, alignment_seq, amr)) in enumerate(zip(toks, poss, alignments, amr_graphs)):

            logger.writeln('Sentence #%d' % (sent_index+1))
            logger.writeln(' '.join(tok_seq))

            amr.setStats(amr_statistics)

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

            (opt_toks, role_toks, node_to_span, edge_to_span, temp_aligned) = extractNodeMapping(alignment_seq, amr)

            temp_unaligned = set(xrange(len(pos_seq))) - temp_aligned

            all_frags = []
            all_alignments = defaultdict(list)

            ####Extract entities mapping#####
            for (frag, wiki_label) in amr.extract_entities():
                if len(opt_toks) == 0:
                    logger.writeln("No alignment for the entity found")

                (aligned_indexes, entity_spans) = all_aligned_spans(frag, opt_toks, role_toks, temp_unaligned)
                root_node = amr.nodes[frag.root]

                entity_mention_toks = root_node.namedEntityMention()
                print 'fragment entity mention: %s' % ' '.join(entity_mention_toks)
                print 'wiki label: %s' % wiki_label

                total_num += 1.0
                if entity_spans:
                    entity_spans = removeRedundant(tok_seq, entity_spans, entity_mention_toks)
                    if len(entity_spans) == 1:
                        singleton_num += 1.0
                        logger.writeln('Single fragment')
                        for (frag_start, frag_end) in entity_spans:
                            logger.writeln(' '.join(tok_seq[frag_start:frag_end]))
                            all_alignments[frag.root].append((frag_start, frag_end, wiki_label))
                    else:
                        multiple_num += 1.0
                        logger.writeln('Multiple fragment')
                        logger.writeln(aligned_indexes)
                        logger.writeln(' '.join([tok_seq[index] for index in aligned_indexes]))

                        for (frag_start, frag_end) in entity_spans:
                            logger.writeln(' '.join(tok_seq[frag_start:frag_end]))
                            all_alignments[frag.root].append((frag_start, frag_end, wiki_label))
                else:
                    empty_num += 1.0
                    _ = all_alignments[frag.root]

            for node_index in node_to_span:
                if node_index in all_alignments:
                    continue

                all_alignments[node_index] = node_to_span[node_index]
                if len(node_to_span[node_index]) > 1:
                    print 'Multiple found:'
                    print amr.nodes[node_index].node_str()
                    for (span_start, span_end, _) in node_to_span[node_index]:
                        print ' '.join(tok_seq[span_start:span_end])

            ##Based on the alignment from node index to spans in the string

            assert len(tok_seq) == len(pos_seq)

            amr_seq, cate_tok_seq, map_seq = categorizeParallelSequences(amr, tok_seq, all_alignments, args.min_prd_freq, args.min_var_freq)
            print >> amrseq_wf, ' '.join(amr_seq)
            print >> tokseq_wf, ' '.join(cate_tok_seq)
            print >> mapseq_wf, '##'.join(map_seq)  #To separate single space

        amrseq_wf.close()
        tokseq_wf.close()
        mapseq_wf.close()

        print "one to one alignment: %lf" % (singleton_num/total_num)
        print "one to multiple alignment: %lf" % (multiple_num/total_num)
        print "one to empty alignment: %lf" % (empty_num/total_num)
    else: #Only build the linearized token sequence

        mle_map = loadMap(args.map_file)
        if args.use_lemma:
            tok_file = os.path.join(args.data_dir, 'lemmatized_token')
        else:
            tok_file = os.path.join(args.data_dir, 'token')

        ner_file = os.path.join(args.data_dir, 'ner')

        all_entities = identify_entities(tok_file, ner_file, mle_map)

        tokseq_result = os.path.join(args.data_dir, 'linearized_tokseq')
        dev_map_file = os.path.join(args.data_dir, 'cate_map')
        tokseq_wf = open(tokseq_result, 'w')
        dev_map_wf = open(dev_map_file, 'w')

        for (sent_index, (tok_seq, pos_seq, entities_in_sent)) in enumerate(zip(toks, poss, all_entities)):
            print 'snt: %d' % sent_index
            n_toks = len(tok_seq)
            aligned_set = set()

            all_spans = []
            #First align multi tokens
            for (start, end, entity_typ) in entities_in_sent:
                if end - start > 1:
                    new_aligned = set(xrange(start, end))
                    aligned_set |= new_aligned
                    entity_name = ' '.join(tok_seq[start:end])
                    if entity_name in mle_map:
                        entity_typ = mle_map[entity_name]
                    else:
                        entity_typ = ('NE_person', "NONE", '-')
                    all_spans.append((start, end, entity_typ))

            #Single token
            for (index, curr_tok) in enumerate(tok_seq):
                if index in aligned_set:
                    continue

                curr_pos = pos_seq[index]
                aligned_set.add(index)

                if curr_tok in mle_map:
                    (category, node_repr, wiki_label) = mle_map[curr_tok]
                    if category.lower() == 'none':
                        all_spans.append((index, index+1, (curr_tok, "NONE", "NONE")))
                    else:
                        all_spans.append((index, index+1, mle_map[curr_tok]))
                else:
                    if curr_tok[0] in '\"\'.':
                        print 'weird token: %s, %s' % (curr_tok, curr_pos)
                        continue
                    if curr_pos[0] == 'V':
                        all_spans.append((index, index+1, ('-VERB-', "NONE", "NONE")))
                    else:
                        all_spans.append((index, index+1, ('-SURF-', "NONE", "NONE")))

            all_spans = sorted(all_spans, key=lambda span: (span[0], span[1]))
            linearized_tokseq, map_repr_seq = getIndexedForm(all_spans)

            print >> tokseq_wf, ' '.join(linearized_tokseq)
            print >> dev_map_wf, '##'.join(map_repr_seq)

        tokseq_wf.close()
        dev_map_wf.close()

#Given the parsed categorized sequence, using the original mapping
#Rebuild the parsed AMR graphs
def replaceResultCategories(args):
    logger.file = open(os.path.join(args.run_dir, 'logger'), 'w')

    tok_file = os.path.join(args.data_dir, 'token')
    lemma_file = os.path.join(args.data_dir, 'lemmatized_token')
    pos_file = os.path.join(args.data_dir, 'pos')
    parsed_file = os.path.join(args.data_dir, 'dev.parsed.1.1.amr')
    ner_file = os.path.join(args.data_dir, 'ner')

    toks = [line.strip().split() for line in open(tok_file, 'r')]
    lemmas = [line.strip().split() for line in open(lemma_file, 'r')]
    poss = [line.strip().split() for line in open(pos_file, 'r')]
    parsed_seqs = [line.strip().split() for line in open(parsed_file, 'r')]

    cate_mle_map = loadMap(args.map_file)  #From token to cate
    node_mle_map = loadMap(args.map_file, 3)  #From token to node repr

    all_entities = identify_entities(lemma_file, ner_file, cate_mle_map)

    replaced_file = os.path.join(args.data_dir, 'replaced_parsed_linear')
    replaced_wf = open(replaced_file, 'w')

    for (sent_index, (tok_seq, lemma_seq, pos_seq, entities_in_sent)) in enumerate(zip(toks, lemmas, poss, all_entities)):
        #print 'snt: %d' % sent_index
        n_toks = len(tok_seq)
        assert len(lemma_seq) == n_toks
        aligned_set = set()

        all_spans = []
        #First align multi tokens
        for (start, end, entity_typ) in entities_in_sent:
            if end - start > 1:
                new_aligned = set(xrange(start, end))
                aligned_set |= new_aligned
                #entity_name = ' '.join(tok_seq[start:end])
                entity_name = ' '.join(lemma_seq[start:end])
                if entity_name in cate_mle_map:
                    entity_typ = cate_mle_map[entity_name]
                else:
                    entity_typ = 'NE_person'
                all_spans.append((start, end, entity_typ))

        #Single token
        #for (index, curr_tok) in enumerate(tok_seq):
        for (index, curr_tok) in enumerate(lemma_seq):
            if index in aligned_set:
                continue

            curr_pos = pos_seq[index]
            aligned_set.add(index)

            if curr_tok in cate_mle_map:
                if cate_mle_map[curr_tok].lower() == 'none':
                    all_spans.append((index, index+1, curr_tok))
                else:
                    all_spans.append((index, index+1, cate_mle_map[curr_tok]))
            else:
                if curr_pos[0] == 'V':
                    all_spans.append((index, index+1, '-VERB-'))
                else:
                    all_spans.append((index, index+1, '-SURF-'))

        all_spans = sorted(all_spans, key=lambda span: (span[0], span[1]))
        #print all_spans

        linearized_tokseq = [l for (start, end, l) in all_spans]
        spans = [(start, end) for (start, end, l) in all_spans]

        linearized_tokseq = getIndexedForm(linearized_tokseq)

        cate_to_node_repr = {}
        #For each category in the linearized tok sequence, replace it with a subgraph repr: linearized amr subgraph
        for (start, end), l in zip(spans, linearized_tokseq):
            if isSpecial(l):
                print start, end, l
                l_nosuffix = re.sub('-[0-9]+', '', l)
                #entity_name = ' '.join(tok_seq[start:end])
                entity_name = ' '.join(lemma_seq[start:end])
                if 'ENT' in l: #Is an entity
                    assert l[:4] == 'ENT_', l
                    node_repr = l_nosuffix[4:]
                    cate_to_node_repr[l] = node_repr
                elif 'NE' in l: #Is a named entity
                    entity_root = l_nosuffix[3:]
                    branch_form = buildLinearEnt(entity_root, tok_seq[start:end])  #Here rebuild the op representation of the named entity
                    cate_to_node_repr[l] = branch_form
                elif 'VERB' in l: #Predicate
                    #assert entity_name in node_mle_map
                    if entity_name in node_mle_map:
                        cate_to_node_repr[l] = node_mle_map[entity_name]
                    else:
                        cate_to_node_repr[l] = '%s-01' % lemma_seq[start]
                elif 'SURF' in l: #Surface form
                    cate_to_node_repr[l] = entity_name
                else: #is a const
                    cate_to_node_repr[l] = entity_name

        parsed_seq = parsed_seqs[sent_index+1]
        parsed_seq = [cate_to_node_repr[l] if l in cate_to_node_repr else l for l in parsed_seq]

        print >> replaced_wf, ' '.join(parsed_seq)
    replaced_wf.close()

def buildLinearEnt(entity_name, ops):
    ops_strs = ['op%d( %s )op%d' % (index, s, index) for (index, s) in enumerate(ops, 1)]
    ent_repr = '%s name( name %s )name' % (entity_name, ' '.join(ops_strs))
    return ent_repr

def isSpecial(symbol):
    for l in ['ENT', 'NE', 'VERB', 'SURF', 'CONST']:
        if l in symbol:
            return True
    return False

def getIndexedForm(linearized_tokseq):
    new_seq = []
    indexer = {}
    map_repr = []
    #cate_to_node_repr = {}
    for (start, end, (tok, node_repr, wiki_label)) in linearized_tokseq:
        if isSpecial(tok):
            new_tok = '%s-%d' % (tok, indexer.setdefault(tok, 0))
            indexer[tok] += 1
            new_seq.append(new_tok)
            #cate_to_node_repr[new_tok] = (node_repr, wiki_label)
            map_repr.append('%s++%s++%s' % (new_tok, node_repr, wiki_label))
        else:
            new_seq.append(tok)
    return new_seq, map_repr

#Given dev or test data, build the linearized token sequence
#Based on entity mapping from training, NER tagger
def conceptID(args):
    return

#Build the entity map for concept identification
#Choose either the most probable category or the most probable node repr
def loadMap(map_file):
    span_to_cate = {}

    #First load all possible mappings each span has
    with open(map_file, 'r') as map_f:
        for line in map_f:
            if line.strip():
                spans = line.strip().split('##')
                for s in spans:
                    try:
                        fields = s.split('++')
                        toks = fields[1]
                        wiki_label = fields[2]
                        node_repr = fields[3]
                        category = fields[-1]
                    except:
                        print spans, line
                        print fields
                        sys.exit(1)
                    if toks not in span_to_cate:
                        span_to_cate[toks] = defaultdict(int)
                    span_to_cate[toks][(category, node_repr, wiki_label)] += 1

    mle_map = {}
    for toks in span_to_cate:
        sorted_types = sorted(span_to_cate[toks].items(), key=lambda x:-x[1])
        mle_map[toks] = sorted_types[0][0]
    return mle_map

#For each sentence, rebuild the map from categorized form to graph side nodes
def rebuildMap(args):
    return

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--amr_file", type=str, help="the original AMR graph files", required=False)
    argparser.add_argument("--stop", type=str, help="stop words file", required=False)
    argparser.add_argument("--lemma", type=str, help="lemma file", required=False)
    argparser.add_argument("--data_dir", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--map_file", type=str, help="map file from training")
    argparser.add_argument("--run_dir", type=str, help="the output directory for saving the constructed forest")
    argparser.add_argument("--use_lemma", action="store_true", help="if use lemmatized tokens")
    argparser.add_argument("--parallel", action="store_true", help="if to linearize parallel sequences")
    argparser.add_argument("--use_stats", action="store_true", help="if use a built-up statistics")
    argparser.add_argument("--stats_dir", type=str, help="the statistics directory")
    argparser.add_argument("--min_prd_freq", type=int, default=50, help="threshold for filtering predicates")
    argparser.add_argument("--min_var_freq", type=int, default=50, help="threshold for filtering non predicate variables")
    argparser.add_argument("--index_unknown", action="store_true", help="if to index the unknown predicates or non predicate variables")

    args = argparser.parse_args()
    replaceResultCategories(args)
    #linearize_amr(args)
    #loadMap('./run_dir/mapseq_noindex')
