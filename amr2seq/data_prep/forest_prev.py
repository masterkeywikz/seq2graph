#!/usr/bin/python
import sys
import time
import pickle
import os
import cPickle
import alignment
import hypergraph
from fragment_hypergraph import FragmentHGNode, FragmentHGEdge
import smatch
from smatch import get_amr_line
import amr_graph
from amr_graph import *
import logger
import gflags
from data_divider import main
import HRGSample
from HRGSample import *
from rule import Rule

FLAGS = gflags.FLAGS

gflags.DEFINE_string(
    'fragment_nonterminal',
    'X',
    'Nonterminal used for phrase forest.')
gflags.DEFINE_bool(
    'delete_unaligned',
    False,
    'Delete unaligned words in phrase decomposition forest.')
gflags.DEFINE_integer(
    'max_type',
    7,
    'Set the maximum attachment nodes each nontermial can have.')

FRAGMENT_NT = '[%s]' % FLAGS.fragment_nonterminal
rule_f = open('train_rules.gr', 'w')
unalign_f = open('unaligned_info', 'w')
#rule_f = open('all_train_tab.gr', 'w')
#unaligned_f =open('all_include.gr', 'w')
def filter_with_maxtype(curr_node):
    root_index = curr_node.frag.root
    ext_set = curr_node.frag.ext_set
    nonterm_type = len(ext_set) if root_index in ext_set else (len(ext_set) + 1)
    #print FLAGS.max_type
    if nonterm_type > FLAGS.max_type:
        #print len(curr_node.frag.ext_set)
        curr_node.set_nosample(True)

#Enlarge a chart with a set of items
def enlarge_chart(prev_chart, new_items):
    for node1 in new_items:
        flag = False
        for node2 in prev_chart:
            if node1.frag == node2.frag:
                for edge in node1.incoming:
                    node2.add_incoming(edge)
                    flag = True

        if not flag: #node1 has never appeared in the previous chart
            prev_chart.add(node1)

#Add one item to a chart
def add_one_item(prev_chart, item):
    flag = False
    for node in prev_chart:
        if node.frag == item.frag:
            for edge in item.incoming:
                node.add_incoming(edge)
                flag = True
    if not flag:
        prev_chart.add(item)

#To verify if a chart item has covered the graph
#i.e. covered all nodes and all edges
def is_goal_item(chart_item):
    fragment = chart_item.frag
    nodes = fragment.nodes
    edges = fragment.edges
    return len(edges) == edges.count()
    #return (len(nodes) == nodes.count()) and (len(edges) == edges.count())

def initialize_edge_alignment(aligned_fragments, edge_alignment):
    for frag in aligned_fragments:
        edge_alignment |= frag.edges

def get_root_amr_node(curr_node, amr_graph):
    curr_node_index = curr_node.frag.root
    curr_amr_node = amr_graph.nodes[curr_node_index]
    return curr_amr_node

#This method output all the unaligned node information of the current AMR graph
def output_all_unaligned_nodes(edge_alignment, amr_graph):
    un_seq = []
    un_nodes = []
    for i in xrange(len(amr_graph.nodes)):
        curr_node = amr_graph.nodes[i]
        c_edge_index = curr_node.c_edge
        if edge_alignment[c_edge_index] == 0: #Found a concept that is not aligned
            un_seq.append(str(curr_node))
            un_nodes.append(curr_node)
            #print >> unalign_f, str(curr_node)
    print >> unalign_f, ' '.join(un_seq)
    return un_nodes

def output_all_unaligned_edges(edge_alignment, amr_graph):
    for i in xrange(len(edge_alignment)):
        if edge_alignment[i] == 0:
            un_seq.append(str(amr_graph.edges[i]))
    print >> unalign_f, ' '.join(un_seq)

#Build one node that would cover unaligned edges going out of a node
def build_one_node(curr_node, amr_graph, edge_alignment):
    curr_node_index = curr_node.frag.root
    #curr_graph_node = amr_graph.nodes[curr_node_index]
    curr_graph_node = get_root_amr_node(curr_node, amr_graph)
    assert edge_alignment[curr_graph_node.c_edge] == 1, 'The edge alignment does not have the aligned node concept'
    unaligned_rels = []
    unaligned_args = []
    unaligned_ops = []
    visited = set()
    if len(curr_graph_node.v_edges) > 0:
        for curr_edge_index in curr_graph_node.v_edges:
            #edge_index = self.edge_dict[edge]
            if curr_edge_index in visited or edge_alignment[curr_edge_index] == 1: #This edge has already been aligned
                continue
            visited.add(curr_edge_index)
            curr_edge = amr_graph.edges[curr_edge_index]
            edge_label = curr_edge.label
            #edge_alignment[curr_edge_index] = 1
            tail_node_index = curr_edge.tail

            if edge_label[:3] == 'ARG' and 'of' not in edge_label:
                unaligned_args.append((curr_edge_index, tail_node_index))
            elif edge_label[:2] == 'op':
                unaligned_ops.append((curr_edge_index, tail_node_index))
            else:
                unaligned_rels.append((curr_edge_index, tail_node_index))

    ext_set = set()
    unaligned_node = None

    used_rels = unaligned_args if len(unaligned_args) > 0 else unaligned_ops if len(unaligned_ops) > 0 else []

    new_node = None
    if len(used_rels) > 0:
        ext_set.add(curr_node_index)
        n_nodes = len(amr_graph.nodes)
        n_edges = len(amr_graph.edges)
        frag = AMRFragment(n_edges, n_nodes, amr_graph)
        frag.set_root(curr_node_index)
        #for rel_index, tail_index in unaligned_rels:
        for rel_index, tail_index in used_rels:
            edge_alignment[rel_index] = 1
            frag.set_edge(rel_index)
            frag.set_node(tail_index)
            ext_set.add(tail_index)
        frag.set_ext_set(ext_set)
        unaligned_node = FragmentHGNode(FRAGMENT_NT, -1, -1, frag, True) #Special here
        new_frag = combine_fragments(curr_node.frag, frag)
        assert new_frag, 'Weird combination found'
        #curr_node.set_nosample(True)

        new_node = FragmentHGNode(FRAGMENT_NT, curr_node.start, curr_node.end, new_frag)

        #children = []
        #children.append(curr_node)
        #children.append(unaligned_node) #This child is a bit special
        #if not check_consist(new_node, children):
        #    print 'inconsistency here'
        #    print str(new_node.frag)
        #    print str(curr_node.frag)
        #    print str(unaligned_node.frag)
        #edge = FragmentHGEdge()
        #edge.add_tail(curr_node)
        #edge.add_tail(unaligned_node)
        #new_node.add_incoming(edge)

        filter_with_maxtype(new_node)

    if not new_node:
        new_node = curr_node

    #Next we try to attach the node's parent
    ret_node = new_node
    if len(curr_graph_node.p_edges) > 0:
        for p_edge in curr_graph_node.p_edges:
            p_amr_edge = amr_graph.edges[p_edge]
            if edge_alignment[p_edge] == 0 and (p_amr_edge.label[:3] != 'ARG' or 'of' in p_amr_edge.label) and p_amr_edge.label[:2] != 'op':
                parent_idx = p_amr_edge.head
                ext_set = set()
                ext_set.add(parent_idx)
                ext_set.add(curr_node_index)
                n_nodes = len(amr_graph.nodes)
                n_edges = len(amr_graph.edges)
                frag = AMRFragment(n_edges, n_nodes, amr_graph)
                frag.set_root(parent_idx)
                edge_alignment[p_edge] = 1
                #frag.set_node(parent_idx)
                frag.set_node(curr_node_index)
                frag.set_edge(p_edge)
                frag.set_ext_set(ext_set)

                unaligned_node = FragmentHGNode(FRAGMENT_NT, -1, -1, frag, True) #Special here
                new_frag = combine_fragments(new_node.frag, frag)
                assert new_frag, 'Another weird combination'
                #if not new_frag:
                #    return new_node
                ret_node = FragmentHGNode(FRAGMENT_NT, new_node.start, new_node.end, new_frag)
                #children = []
                #children.append(new_node)
                #children.append(unaligned_node) #This child is a bit special
                #if not check_consist(ret_node, children):
                #    print 'inconsistency here'
                #    print str(ret_node.frag)
                #    print str(new_node.frag)
                #    print str(unaligned_node.frag)
                #edge = FragmentHGEdge()
                #edge.add_tail(new_node)
                #edge.add_tail(unaligned_node)
                #ret_node.add_incoming(edge)
                return ret_node

    return ret_node

# extract all the binarized fragments combinations for AMR
# each chart item is a set of fragments are consistent with a span of aligned strings
def fragment_decomposition_forest(fragments, amr_graph):
    # save the index mapping so that we can restore indices after phrase
    # decomposition forest generation

    edge_alignment = bitarray(len(amr_graph.edges))
    if edge_alignment.count() != 0:
        edge_alignment ^= edge_alignment

    n = len(fragments) #These fragments are aligned, and have some order based on the strings

    chart = [[set() for j in range(n+1)] for i in range(n+1)]

    initialize_edge_alignment(fragments, edge_alignment)

    start_time = time.time()
    #The leaves of the forest is identified concept fragments
    for i in xrange(n):
        j = i + 1
        frag = fragments[i]
        curr_node = FragmentHGNode(FRAGMENT_NT, i, j, frag)

        #s = Sample(hypergraph.Hypergraph(curr_node), 0)
        #curr_node.cut = 1
        #new_rule, _ = s.extract_one_rule(curr_node, None, list(curr_node.frag.ext_set))
        #rule_f.write('%s\n' % new_rule.dumped_format())

        new_node = build_one_node(curr_node, amr_graph, edge_alignment)
        filter_with_maxtype(new_node)
        chart[i][j].add(new_node)

    unaligned_fragments = amr_graph.extract_unaligned_fragments(edge_alignment)
    #logger.writeln('finished extracting unaligned')
    unaligned_nodes = []
    for unaligned_frag in unaligned_fragments:
        unaligned_node = FragmentHGNode(FRAGMENT_NT, -1, -1, unaligned_frag, True) #Special here
        unaligned_nodes.append(unaligned_node)

    #logger.writeln('Begin dealing with unary')
    for i in xrange(n):
        j = i + 1
        curr_candidate = chart[i][j]
        updated = True
        while updated:
            updated = False
            new_node_set = set()
            curr_time = time.time()
            if curr_time - start_time > 30:
                return None
            for node1 in curr_candidate:
                for unaligned_node in unaligned_nodes:
                    #Before combine two fragments, check if they are disjoint and adjacent
                    if check_disjoint(node1.frag, unaligned_node.frag) and check_adjacent(node1.frag, unaligned_node.frag):
                        new_frag = combine_fragments(node1.frag, unaligned_node.frag)
                        if new_frag is None:
                            continue
                        #node1.set_nosample(True)
                        new_node = FragmentHGNode(FRAGMENT_NT, i, j, new_frag)

                        edge = FragmentHGEdge()
                        edge.add_tail(node1)
                        edge.add_tail(unaligned_node)
                        new_node.add_incoming(edge)
                        #s = Sample(hypergraph.Hypergraph(new_node), 0)
                        #new_node.cut = 1
                        #new_rule, _ = s.extract_one_rule(new_node, None, list(new_node.frag.ext_set))
                        #rule_f.write('plus unaligned:\n%s\n' % new_rule.dumped_format())
                        updated = True
                        filter_with_maxtype(new_node)
                        add_one_item(new_node_set, new_node)
            if updated:
                enlarge_chart(chart[i][j], new_node_set)
                curr_candidate = new_node_set

    #logger.writeln('Finished dealing with unary')
    for span in xrange(2, n+1):
        for i in xrange(0, n):
            j = i + span
            if j > n:
                continue
            for k in xrange(i+1, j):
                if len(chart[i][k]) == 0 or len(chart[k][j]) == 0:
                    continue
                for node1 in chart[i][k]:
                    for node2 in chart[k][j]:
                        curr_time = time.time()
                        if curr_time - start_time > 30:
                            return None
                    #Before combine two fragments, check if they are disjoint and adjacent
                        if check_disjoint(node1.frag, node2.frag) and check_adjacent(node1.frag, node2.frag):
                            new_frag = combine_fragments(node1.frag, node2.frag)
                            if new_frag is None:
                                continue
                            #logger.writeln(str(new_frag))
                            #logger.writeln(new_frag.ext_nodes_str())
                            new_node = FragmentHGNode(FRAGMENT_NT, i, j, new_frag)


                            children = []
                            children.append(node1)
                            children.append(node2)
                            if not check_consist(new_node, children):
                                print 'inconsistency here'
                                print str(new_node.frag)
                                print str(node1.frag)
                                print str(node2.frag)
                            edge = FragmentHGEdge()
                            edge.add_tail(node1)
                            edge.add_tail(node2)
                            new_node.add_incoming(edge)
                            #s = Sample(hypergraph.Hypergraph(new_node), 0)
                            #new_node.cut = 1
                            #new_rule, _ = s.extract_one_rule(new_node, None, list(new_node.frag.ext_set))
                            #rule_f.write('%s\n' % new_rule.dumped_format())
                            filter_with_maxtype(new_node)
                            add_one_item(chart[i][j], new_node)

    if chart[0][n] is None:
        #rule_f.write('\n')
        print '##################################'
        print 'The goal chart is empty, fail to build a goal item'
        print 'Alignment fragments:'
        for frag in fragments:
            print str(frag)
        print 'Unaligned fragments:'
        for frag in unaligned_fragments:
            print str(frag)
        print '#################################'
        return None

    #rule_f.write('\n')
    hg = None
    for node in chart[0][n]:
        #print str(node)
        if is_goal_item(node):
            hg = hypergraph.Hypergraph(node)
            assert len(node.frag.ext_set) == 0, 'The whole graph should not have external nodes: %s \n %s' %(str(node.frag), node.frag.ext_nodes_str())
            #print str(node.frag)

    #assert hg is not None, 'Failed to build a goal item'
    if hg is None:
        print '##################################'
        print 'No goal item in the final chart'
        print 'Alignment fragments:'
        for frag in fragments:
            print str(frag)
        print 'Unaligned fragments:'
        for frag in unaligned_fragments:
            print str(frag)
        print '#################################'
        sys.stdout.flush()
        return None
    #hg.assert_done('topo_sort')
    #print type(hg)
    return hg

def parallel_forest_construct(argv):
    if argv[1] == '-m': #The main processor
        main(argv[1:4], 5000)
        data_dir = argv[3]
        file_no = int(argv[4])
        forest_dir = argv[5]
        cluster_nodes = argv[6].split('+')  #The cluster nodes are separted with '+'
        assert len(cluster_nodes) == file_no, 'The number of files and number of cluster nodes does not match'
        FLAGS.max_type = int(argv[7])

        os.system('rm -rf %s' % forest_dir)
        os.mkdir(forest_dir)
        i = 0
        for curr_node in cluster_nodes: #iterate through each file
            data_file = 'graph_%d' % i
            align_file = 'align_%d' % i
            sent_file = 'sent_%d' % i
            des_file = 'forest_%d' % i
            used_sent_file = 'used_sent_%d' % i

            data_file = os.path.join(data_dir, data_file)
            align_file = os.path.join(data_dir, align_file)
            error_log_file = 'error_log_%d' % i
            sent_file = os.path.join(data_dir, sent_file)
            used_sent_file = os.path.join(forest_dir, used_sent_file)
            des_file = os.path.join(forest_dir, des_file)
            print 'start to launch program in %s' % curr_node

            cmd = 'python %s -s %s %s %s %s %s %d %d' % (argv[0], data_file, align_file, sent_file, des_file, used_sent_file, i, FLAGS.max_type)
            os.system(r'ssh %s "cd %s; nohup %s >& %s" &' % (curr_node, os.getcwd(), cmd, error_log_file))
            i += 1
            if i >= file_no:
                break

    else:
        #assert len(argv) == 8, 'There should 8 arguments for the slaves'
        subset_id = int(argv[7])
        #print argv
        logger.file = open('logger_%d' % subset_id, 'w')
        #print argv
        amr_graph_file = argv[2]
        align_file = argv[3]
        sent_file = argv[4]
        des_file = argv[5]
        used_sent_file = argv[6]
        FLAGS.max_type = int(argv[8])
        #logger.writeln(argv[0])
        #logger.writeln(argv[1])

        f1 = open(amr_graph_file, 'rb')
        amr_graphs = cPickle.load(f1)
        f1.close()

        f2 = open(align_file, 'r')
        sent_alignments = []
        alignment_line = f2.readline()
        while alignment_line != '':
            alignment_line = alignment_line.strip()
            sent_alignments.append(alignment_line)
            alignment_line = f2.readline()
        f2.close()

        f3 = open(sent_file, 'r')
        sents = []
        sent_line = f3.readline()
        while sent_line != '':
            sent_line = sent_line.strip()
            sents.append(sent_line)
            sent_line = f3.readline()
        f3.close()

        sent_no = len(sents)
        assert sent_no == len(sent_alignments) and sent_no == len(amr_graphs), '%d %d %d' % (sent_no, len(sent_alignments), len(amr_graphs))

        frag_forests = []
        sent_indexes = []


        #for sent_index in xrange(sent_no):
        num_self_cycle = 0
        for sent_index in xrange(sent_no):
        #for sent_index in xrange(30):
            curr_sent_index = 500 * subset_id + sent_index
            amr_graph = amr_graphs[sent_index]
            if amr_graph.check_self_cycle():
                num_self_cycle += 1
                logger.writeln('The %d-th sentence has self cycle' % curr_sent_index)
                #logger.writeln(str(amr_graph))
                #continue
            #logger.writeln('There are %d self cycle sentences' % num_self_cycle)
            logger.writeln(str(amr_graph))
            alignment_line = sent_alignments[sent_index]
            if alignment_line == '':
                logger.writeln('The %d-th sentence is totally unaligned' % curr_sent_index)
                logger.writeln(str(amr_graph))
                continue
            sent = sents[sent_index]
            amr_graph.set_sentence(sent.strip().split())

            alignments = alignment_line.strip().split()
            alignments = [tuple(align.split('|')) for align in alignments]
            aligned_fragments = []
            error_happened = False

            aligned_words = set()
            for align in alignments:
                s_side = align[0]
                f_side = align[1]
                s_start = s_side.split('-')[0]
                s_start = (int)(s_start)
                s_end = s_side.split('-')[1]
                s_end = (int)(s_end)
                fragment = amr_graph.retrieve_fragment(f_side)
                if fragment is None:
                    error_happened = True
                    break
                fragment.set_span(s_start, s_end)
                aligned_fragments.append((s_start, s_end, fragment))
                aligned_words |= set(range(s_start, s_end))

            if error_happened:
                logger.writeln('The %d-th sentence has wrong alignments' % curr_sent_index)
                logger.writeln(str(amr_graph))
                continue

            aligned_fragments = sorted(aligned_fragments) #This sort this used to constrain the possible combinations

            curr_start = 0
            curr_end = 0
            unaligned_wset = set()
            for start, end, frag in aligned_fragments:
                if start < curr_end:
                    error_happened = True
                    break
                unaligned_wset |= set(xrange(curr_end, start))
                curr_end = end

            if error_happened:
                continue

            if curr_end != len(amr_graph.sent):
                unaligned_wset |= set(xrange(curr_end, len(amr_graph.sent)))

            aligned_fragments = [z for x,y,z in aligned_fragments]

            logger.writeln('Aligned fragments:')
            for frag in aligned_fragments:
                logger.writeln('%s  %s' % (' '.join(frag.str_list()), str(frag)))

            #Print the unaligned info to file
            print >> unalign_f, 'sent %d:' % sent_index
            toks = sent.split()
            unaligned_words = set(range(len(toks))) - aligned_words
            un_seq = []
            for pos in unaligned_words:
                un_seq.append(toks[pos])
                #print >> unalign_f, toks[pos]
            print >> unalign_f, ' '.join(un_seq)

            hg = fragment_decomposition_forest(aligned_fragments, amr_graph)
            if hg is not None:
                hg.set_unaligned(unaligned_wset)
                frag_forests.append(hg)
                sent_indexes.append(curr_sent_index)
            logger.writeln('Finished sentence %d.' % curr_sent_index)

        f4 = open(des_file, 'wb')
        cPickle.dump(frag_forests, f4)
        f4.close()
        f5 = open(used_sent_file, 'wb')
        cPickle.dump(sent_indexes, f5)
        f5.close()
        logger.writeln('Total used sentences: %d' % len(sent_indexes))
        logger.writeln('finished')

if __name__ == '__main__':
    amr_file = sys.argv[1]
    sentence_file = sys.argv[2]
    alignment_file = sys.argv[3]
    #frag_forests = initialize_fragment_forest(amr_file, sentence_file, alignment_file)
    parallel_forest_construct(sys.argv)
