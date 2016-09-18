#!/usr/bin/python
import sys
import time
import pickle
import os
import cPickle
import alignment
import hypergraph
from fragment_hypergraph import FragmentHGNode, FragmentHGEdge
from amr_graph import *
from amr_utils import *
import logger
import gflags
from HRGSample import *
from rule import Rule, retrieve_edges
import argparse
from re_utils import *
import re
from collections import defaultdict
from filter_stop_words import *
from lemma_util import initialize_lemma
from feature import *
from entities import load_entities, identify_entities
from date_extraction import *
from copy import copy
FLAGS = gflags.FLAGS

gflags.DEFINE_string(
    'fragment_nonterminal',
    'X',
    'Nonterminal used for phrase forest.')
gflags.DEFINE_bool(
    'delete_unaligned',
    False,
    'Delete unaligned words in phrase decomposition forest.')
gflags.DEFINE_bool(
    'href',
    False,
    'Delete unaligned words in phrase decomposition forest.')
gflags.DEFINE_integer(
    'max_type',
    7,
    'Set the maximum attachment nodes each nontermial can have.')

FRAGMENT_NT = '[%s]' % FLAGS.fragment_nonterminal
def read_toks(filename):
    tok_seqs = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                toks = line.split()
                tok_seqs.append(toks)
    return tok_seqs

def build_one_entity(toks, frag_label):
    entity_label = frag_label.split('+')[-1]
    result_s = '. :%s/%s' % (entity_label[0], entity_label)
    result_s += ' :name (. :n/name'
    for (i, tok) in enumerate(toks):
        result_s += ' :op%d (. :"%s" )' % (i+1, tok)
    result_s += ')'
    result_s += ' :wiki (. :-)'
    return '(%s)' % result_s
    #return "(. :%s %s)" % (frag_label.split('+')[-1], ' '.join(toks))

def build_one_predicate(curr_pred, frag_label):
    re_pat = re.compile('\.:ARG[0-9]-of')
    match = re_pat.search(frag_label, 0)
    ArgPart = frag_label
    Argof = None
    result_s = ''
    if match: #There is a arg-of relation
        Argof = match.group(0)
        ArgPart = frag_label.replace(Argof, '')
        result_s = '. %s ' % Argof[1:]

    Arg_s = '.'
    arg_pat = re.compile('(:ARG[0-9])\.')
    position = 0
    index = 0
    while position < len(ArgPart):
        match = arg_pat.search(ArgPart, position)
        if not match:
            break
        match_s = match.group(1)
        Arg_s += ' %s .*%d' % (match_s, index)
        index += 1
        position = match.end()
    Arg_s += ' :%s' % curr_pred

    if Argof:
        return ('(%s (%s ))' % (result_s, Arg_s), index+1, '0' + '0'*index)
    else:
        return ('(%s )' % Arg_s, index+1, '1'+ '0'* index)
    return None

def read_sentence(input_f):
    sent_map = []
    for line in input_f:
        line = line.strip()
        if line:
            fields = line.split()
            frag_label = fields[0]
            span = [int(x) for x in fields[1].split('-')]
            #type_ident = [int(x) for x in fields[1:]]
            sent_map.append(tuple([frag_label] + span))
        else:
            if len(sent_map) == 0:
                continue
            return sent_map
    return []

def load_mappings(stats_dir):
    word2pred_files = os.popen('ls %s/word2pred_*' % stats_dir).read().split()
    ent2frag_files = os.popen('ls %s/ent2frag_*' % stats_dir).read().split()
    label2frag_files = os.popen('ls %s/label2frag_*' % stats_dir).read().split()

    #assert len(stats_files) == 8
    word2predicate = {}
    ent2frag = {}
    label2frag = {}

    stop_words = set([line.strip() for line in open('./stop_words', 'r')])
    non_map_words = set(['am', 'is', 'are', 'be', 'a', 'an', 'the'])

    unknown_set = set()

    for file in word2pred_files:
        f = open(file, 'rb')
        stats = cPickle.load(f)

        for curr_tok in stats:
            if curr_tok not in word2predicate:
                word2predicate[curr_tok] = defaultdict(int)
            for pred in stats[curr_tok]:
                word2predicate[curr_tok][pred] += stats[curr_tok][pred]

    word2most = {}
    for tok in word2predicate:
        pred2freq = sorted(word2predicate[tok].items(), key=lambda item: (item[1], item[0]))
        word2most[tok] = pred2freq[-1][0]

    for file in ent2frag_files:

        f = open(file, 'rb')
        stats = cPickle.load(f)

        for ent in stats:
            ent2frag[ent] = stats[ent]

    for file in label2frag_files:
        f = open(file, 'rb')
        stats = cPickle.load(f)

        for label in stats:
            label2frag[label] = stats[label]

    return (word2most, ent2frag, label2frag)

def preprocess_tok(toks, aligned_toks):
    for (i, tok) in enumerate(toks):
        if not has_letter(tok):
            continue

        if i in aligned_toks:
            continue

        while toks[i][-1] in ",.)(:\"\'" and len(toks[i]) > 1:
            toks[i] = toks[i][:-1]

def has_letter(tok):
    match = re.search('[A-Za-z0-9]+', tok)
    if match:
        return True
    return False

def extract_li(tok):  #Only for first tok
    if len(tok) == 2: #1. 2.
        if tok[1] == '.' and is_num(tok[0]):
            return (True, '[A1-0]', '(. :li (. :x/%s ))' % tok[0])
    return (False, None, None)

def concept_id(args):

    tok_seqs = read_toks(args.tok_file)
    lemma_seqs = read_toks(args.lemma_file)
    pos_seqs = read_toks(args.pos_file)

    #The process to extract entities
    train_entity_set = load_entities(args.stats_dir)
    all_entities = identify_entities(args.tok_file, args.ner_file, train_entity_set)
    all_dates = extract_all_dates(args.tok_file)

    non_map_words = set(['am', 'is', 'are', 'be', 'a', 'an', 'the', ',', '.', '..', '...', ':', '(', ')', '@-@', 'there', 'they', 'do', 'and', '\"', '-@' ])

    special_words = set(['a', 'the', 'an', 'it', 'its', 'are', 'been', 'have', 'has', 'had'])
    begin_words = set(['if', 'it'])

    tok_map = {}
    lemma_map = {}
    tok_f = open(args.tok_map, 'rb')
    tok_map = cPickle.load(tok_f)
    tok_f.close()

    lemma_f = open(args.lemma_map, 'rb')
    lemma_map = cPickle.load(lemma_f)
    lemma_f.close()

    new_tok_f = open('%s.temp' % args.tok_file, 'w')
    new_lemma_f = open('%s.temp' % args.lemma_file, 'w')

    result_f = open(args.output, 'w')

    der_lemma_map = initialize_lemma('./lemmas/der.lemma')

    for (i, (toks, lemmas, poss)) in enumerate(zip(tok_seqs, lemma_seqs, pos_seqs)):
        #print 'sentence %d' % i
        n_toks = len(toks)
        visited = set()

        orig_toks = copy(toks)
        toks = [t.lower() for t in toks]
        lemmas = [t.lower() for t in lemmas]

        ent_in_sent = all_entities[i]
        dates_in_line = all_dates[i]
        assert len(toks) == len(lemmas)
        assert len(toks) == len(poss)

        aligned_toks = set()  #See which position is covered
        aligned_rules = []

        for (start, end, lhs, frag_str) in dates_in_line:
            new_aligned = set(xrange(start, end))
            aligned_toks |= new_aligned

            rule_str = build_rule_str(lhs, toks, start, end, frag_str)
            aligned_rules.append(rule_str)

        if 0 not in aligned_toks:
            (is_li, lhs, frag_str) = extract_li(toks[0])
            if is_li:
                aligned_toks.add(0)
                rule_str = build_rule_str(lhs, toks, 0, 1, frag_str)
                aligned_rules.append(rule_str)

        for start in xrange(n_toks):
            (has_slash, lhs, frag_str) = dealwith_slash(toks[start:start+1], tok_map)
            if has_slash:
                aligned_toks.add(start)
                rule_str = build_rule_str(lhs, toks, start, start+1, frag_str)
                aligned_rules.append(rule_str)
                print 'Retrieved slash'
                print '%s : %s' % (toks[start], rule_str)
                sys.stdout.flush()

        #Dealing with entities
        for (start, end, entity_typ) in ent_in_sent:

            new_aligned = set(xrange(start, end))
            if len(new_aligned & aligned_toks) != 0:
                continue

            aligned_toks |= new_aligned

            curr_ent = '_'.join(toks[start:end])
            curr_lex = ' '.join(toks[start:end])
            if curr_lex in tok_map:
                items = tok_map[curr_lex][''].items()
                assert len(items) > 0
                items = sorted(items, key=lambda it: it[1])

                (lhs, frag_part) = items[-1][0]
                if lhs.strip() == 'Nothing':
                    continue

                rule_str = '%d-%d####%s ## %s ## %s' % (start, end, lhs, ' '.join(toks[start:end]), frag_part)
                aligned_rules.append(rule_str)
            else:
                if not entity_typ:
                    continue
                assert entity_typ
                #if 'PER' in entity_typ: #Identified as a person
                frag_label = 'entity+person'
                if 'PER' in entity_typ:
                    frag_label = 'entity+person'
                elif 'ORG' in entity_typ:
                    frag_label = 'entity+organization'
                elif 'LOC' in entity_typ:
                    frag_label = 'entity+city'

                frag_str = build_one_entity(orig_toks[start:end], frag_label)
                rule_str = '%d-%d####[A1-1] ## %s ## %s' % (start, end, ' '.join(toks[start:end]), frag_str)
                aligned_rules.append(rule_str)

        #continue
        preprocess_tok(toks, aligned_toks)
        preprocess_tok(lemmas, aligned_toks)

        for temp_t in toks:
            assert len(temp_t.strip()) > 0

        print >>new_tok_f, ' '.join(toks)
        print >>new_lemma_f, ' '.join(lemmas)

        possible_aligned = set()
        for start in xrange(n_toks):
            if start in visited:
                continue

            for length in xrange(n_toks+1, 0, -1):
                end = start + length

                if end > n_toks:
                    continue

                span_set = set(xrange(start, end))
                if len(span_set & aligned_toks) != 0:
                    continue

                if toks[start] in non_map_words and end-start > 1:
                    continue

                #if length >= 2:
                #    if poss[end-1] == 'IN' or poss[end-1] == 'DT':

                seq_str = ' '.join(toks[start:end])
                lem_seq_str = ' '.join(lemmas[start:end])
                if length == 1 and seq_str in non_map_words:  #Should verify if these two lines should be commented
                    continue

                contexts = get_context(toks, lemmas, poss, start, end)
                aligned_value = find_maps(seq_str, lem_seq_str, tok_map, lemma_map, contexts)
                if aligned_value:
                    (lhs, frag_part) = aligned_value
                    lhs = lhs.strip()
                    frag_part = frag_part.strip()
                    if lhs == 'Nothing':
                        continue

                    rule_str = build_rule_str(lhs, toks, start, end, frag_part)
                    if length >= 2 and length <= 4:
                        if toks[end-1] in special_words:
                            print 'discarded: %s' % rule_str
                            sys.stdout.flush()
                            continue

                        if toks[start] in begin_words:
                            print 'begin discard: %s' % rule_str
                            sys.stdout.flush()
                            continue

                    aligned_rules.append(rule_str)
                    #if poss[end-1] == 'the' or poss[end-1] == 'DT' or poss[start] == 'IN' or poss[start] == 'DT':
                    #    print rule_str
                    #    sys.stdout.flush()

                    if end - start == 1:
                        aligned_toks.add(start)

                    if num_nonmap(non_map_words, toks[start:end]) > 1:
                        if len(possible_aligned & span_set) == 0:
                            #new_aligned = set(xrange(start, end))
                            aligned_toks |= span_set
                            possible_aligned |= span_set
                            break

                    possible_aligned |= span_set

        unaligned_toks = set(xrange(n_toks)) - aligned_toks
        retrieve_unaligned(unaligned_toks, toks, lemmas, poss, der_lemma_map, aligned_rules, non_map_words, tok_map, lemma_map)

        print >>result_f, '%s ||| %s ||| %s' % (' '.join(toks), ' '.join([str(k) for k in unaligned_toks]), '++'.join(aligned_rules))
        #print ' '.join(['%s/%s/%s' % (toks[k], lemmas[k], poss[k]) for k in unaligned_toks])
    new_tok_f.close()
    new_lemma_f.close()
    result_f.close()

def num_nonmap(non_map_words, toks):
    n = 0
    for tok in toks:
        if tok not in non_map_words:
            n += 1
    return n

def is_num(s):
    regex = re.compile('[0-9]+([^0-9\s]*)')
    match = regex.match(s)
    return match and len(match.group(1)) == 0

def retrieve_unaligned(unaligned_toks, toks, lemmas, poss, der_lemma_map, aligned_rules, non_map_words, tok_map, lemma_map):

    #curr_non_map_words = ['off', 'in', 'of', 'between', 'and', 'must', 'have']
    #Retrieve numbers
    for index in unaligned_toks.copy():
        if is_num(toks[index]):
            rule_str = build_rule_str('[A1-1]', toks, index, index+1, '(. :%s )' % toks[index])
            aligned_rules.append(rule_str)
            unaligned_toks.remove(index)
            print 'Retrieved: %s' % toks[index]

    for index in unaligned_toks.copy():
        curr_tok = toks[index]
        curr_lem = lemmas[index]
        if curr_tok in non_map_words or len(curr_tok) <= 1:
            continue

        if not has_letter(curr_tok):
            continue

        #Already checked
        if curr_tok in tok_map or curr_lem in lemma_map:
            continue

        if poss[index] == 'NN':
            if curr_tok in der_lemma_map:
                der_lem = list(der_lemma_map[curr_tok])[0]
                assert der_lem != curr_tok
                contexts = get_context(toks, lemmas, poss, index, index+1)
                aligned_value = find_maps(der_lem, der_lem, tok_map, lemma_map, contexts)
                if aligned_value:
                    (lhs, frag_part) = aligned_value
                    lhs = lhs.strip()
                    frag_part = frag_part.strip()
                    if lhs == 'Nothing':
                        continue
                    else:
                        rule_str = build_rule_str(lhs, toks, index, index+1, frag_part)
                        aligned_rules.append(rule_str)
                        unaligned_toks.remove(index)
                        print 'Retrieved: %s: %s' % (toks[index], der_lem)

            elif (not curr_tok.endswith('ion')) and (not curr_tok.endswith('ist')) and (not curr_tok.endswith('er')):
                rule_str = build_rule_str('[A1-1]', toks, index, index+1, '(. :%s/%s )' % (curr_tok[0], curr_tok))
                aligned_rules.append(rule_str)
                unaligned_toks.remove(index)
                print 'Retrieved: %s' % toks[index]

        elif poss[index] == 'NNS':
            if curr_lem in der_lemma_map:
                der_lem = list(der_lemma_map[curr_lem])[0]
                contexts = get_context(toks, lemmas, poss, index, index+1)
                aligned_value = find_maps(der_lem, der_lem, tok_map, lemma_map, contexts)
                if aligned_value:
                    (lhs, frag_part) = aligned_value
                    lhs = lhs.strip()
                    frag_part = frag_part.strip()
                    if lhs == 'Nothing':
                        continue
                    else:
                        rule_str = build_rule_str(lhs, toks, index, index+1, frag_part)
                        aligned_rules.append(rule_str)
                        unaligned_toks.remove(index)
                        print 'Retrieved: %s: %s' % (toks[index], der_lem)

            elif (not curr_lem.endswith('ion')) and (not curr_lem.endswith('ist')) and (not curr_lem.endswith('er')):
                rule_str = build_rule_str('[A1-1]', toks, index, index+1, '(. :%s/%s )' % (curr_lem[0], curr_lem))
                aligned_rules.append(rule_str)
                unaligned_toks.remove(index)
                print 'Retrieved: %s: %s' % (toks[index], curr_lem)
        elif poss[index].startswith('NNP'):
            #if poss[index] == 'NNPS' and curr_tok[-1] == 's':
            #    curr_tok = curr_tok[:-1]

            if curr_lem in der_lemma_map:
                der_lem = list(der_lemma_map[curr_lem])[0]
                contexts = get_context(toks, lemmas, poss, index, index+1)
                aligned_value = find_maps(der_lem, der_lem, tok_map, lemma_map, contexts)
                if aligned_value:
                    (lhs, frag_part) = aligned_value
                    lhs = lhs.strip()
                    frag_part = frag_part.strip()
                    if lhs == 'Nothing':
                        continue
                    else:
                        rule_str = build_rule_str(lhs, toks, index, index+1, frag_part)
                        aligned_rules.append(rule_str)
                        unaligned_toks.remove(index)
                        print 'Retrieved: %s: %s' % (toks[index], der_lem)

def build_rule_str(lhs, toks, start, end, frag_str):
    return '%d-%d####%s ## %s ## %s' % (start, end, lhs, ' '.join(toks[start:end]), frag_str)

def get_groups(contexts):
    word_groups = []
    word_groups.append((contexts[0], contexts[1], contexts[2], contexts[3]))
    word_groups.append(((contexts[0], contexts[1]), contexts[2]))
    word_groups.append((contexts[1], (contexts[2], contexts[3])))
    word_groups.append((contexts[1], contexts[2]))
    word_groups.append((contexts[0], contexts[1]))
    word_groups.append((contexts[2], contexts[3]))

    lemma_groups = []
    lemma_groups.append((contexts[4], contexts[5], contexts[6], contexts[7]))
    lemma_groups.append(((contexts[4], contexts[5]), contexts[6]))
    lemma_groups.append((contexts[5], (contexts[6], contexts[7])))
    lemma_groups.append((contexts[5], contexts[6]))
    lemma_groups.append((contexts[4], contexts[5]))
    lemma_groups.append((contexts[6], contexts[7]))

    second_word_groups = []
    second_word_groups.append((contexts[1], ''))
    second_word_groups.append(('', contexts[2]))

    second_lemma_groups = []
    second_lemma_groups.append((contexts[5], ''))
    second_lemma_groups.append(('', contexts[6]))

    pos_tag_groups = []
    pos_tag_groups.append((contexts[8], contexts[9], contexts[10], contexts[11]))
    pos_tag_groups.append(((contexts[8], contexts[9]), contexts[10]))
    pos_tag_groups.append((contexts[9], (contexts[10], contexts[11])))
    pos_tag_groups.append((contexts[9], contexts[10]))
    pos_tag_groups.append((contexts[8], contexts[9]))
    pos_tag_groups.append((contexts[10], contexts[11]))
    pos_tag_groups.append((contexts[9], ''))
    pos_tag_groups.append(('', contexts[10]))

    return (word_groups, lemma_groups, second_word_groups, second_lemma_groups, pos_tag_groups)

def top_item(count_map):
    items = count_map.items()
    assert len(items) > 0
    items = sorted(items, key=lambda it: it[1])
    return items[-1][0]

def is_entity(item):
    (lhs, frag_str) = item
    if ':name' in frag_str and 'op' in frag_str:
        return True
    return False

def wrong_ext(item):
    (lhs, frag_str) = item
    if lhs.strip() == 'Nothing':
        return False

    type = int(lhs[2])
    if len(frag_str.split('.*')) != type:
        return True
    return False

def find_maps(seq_str, lem_seq_str, tok_map, lemma_map, contexts):
    if seq_str in tok_map or lem_seq_str in lemma_map:
        (word_groups, lemma_groups, second_word_groups, second_lemma_groups, pos_tag_groups) = get_groups(contexts)

    if seq_str in tok_map:
        for ctx_typ in word_groups:
            if ctx_typ in tok_map[seq_str]:
                curr_item = top_item(tok_map[seq_str][ctx_typ])
                if not (is_entity(curr_item) or wrong_ext(curr_item)):
                    return curr_item

        for ctx_typ in lemma_groups:
            if ctx_typ in tok_map[seq_str]:
                curr_item = top_item(tok_map[seq_str][ctx_typ])
                if not (is_entity(curr_item) or wrong_ext(curr_item)):
                    return curr_item

        for ctx_typ in second_word_groups:
            if ctx_typ in tok_map[seq_str]:
                curr_item = top_item(tok_map[seq_str][ctx_typ])
                if not (is_entity(curr_item) or wrong_ext(curr_item)):
                    return curr_item

        for ctx_typ in second_lemma_groups:
            if ctx_typ in tok_map[seq_str]:
                curr_item = top_item(tok_map[seq_str][ctx_typ])
                if not (is_entity(curr_item) or wrong_ext(curr_item)):
                    return curr_item

    if lem_seq_str in lemma_map:
        for ctx_typ in word_groups:
            if ctx_typ in lemma_map[lem_seq_str]:
                curr_item = top_item(lemma_map[lem_seq_str][ctx_typ])
                if not (is_entity(curr_item) or wrong_ext(curr_item)):
                    return curr_item

        for ctx_typ in lemma_groups:
            if ctx_typ in lemma_map[lem_seq_str]:
                curr_item = top_item(lemma_map[lem_seq_str][ctx_typ])
                if not (is_entity(curr_item) or wrong_ext(curr_item)):
                    return curr_item

        for ctx_typ in second_word_groups:
            if ctx_typ in lemma_map[lem_seq_str]:
                curr_item = top_item(lemma_map[lem_seq_str][ctx_typ])
                if not (is_entity(curr_item) or wrong_ext(curr_item)):
                    return curr_item

        for ctx_typ in second_lemma_groups:
            if ctx_typ in lemma_map[lem_seq_str]:
                curr_item = top_item(lemma_map[lem_seq_str][ctx_typ])
                if not (is_entity(curr_item) or wrong_ext(curr_item)):
                    return curr_item

    if seq_str in tok_map:
        for ctx_typ in pos_tag_groups:
            if ctx_typ in tok_map[seq_str]:
                curr_item = top_item(tok_map[seq_str][ctx_typ])
                if not (is_entity(curr_item) or wrong_ext(curr_item)):
                    return curr_item

        assert '' in tok_map[seq_str]
        curr_item = top_item(tok_map[seq_str][''])
        if not (is_entity(curr_item) or wrong_ext(curr_item)):
            return curr_item

    if lem_seq_str in lemma_map:
        for ctx_typ in pos_tag_groups:
            if ctx_typ in lemma_map[lem_seq_str]:
                curr_item = top_item(lemma_map[lem_seq_str][ctx_typ])
                if not (is_entity(curr_item) or wrong_ext(curr_item)):
                    return curr_item

        assert '' in lemma_map[lem_seq_str]
        curr_item = top_item(lemma_map[lem_seq_str][''])
        if not (is_entity(curr_item) or wrong_ext(curr_item)):
            return curr_item

    return None

def retrieve_map(tok, tok_map):
    if tok not in tok_map:
        return (False, None)

    items = tok_map[tok][''].items()
    assert len(items) > 0
    items = sorted(items, key=lambda it: it[1])

    (lhs, frag_part) = items[-1][0]
    if lhs.strip() == 'Nothing':
        return (False, None)

    if lhs.strip() != '[A1-1]':
        return (False, None)

    return (True, frag_part)

def dealwith_slash(toks, tok_map):
    if len(toks) == 1 and has_letter(toks[0]):

        #Get rid of leading and ending slash
        while toks[0].endswith('/') and len(toks[0]) > 1:
            toks[0] = toks[0][:-1]

        while toks[0].startswith('/') and len(toks[0]) > 1:
            toks[0] = toks[0][1:]

        if '/' in toks[0]: #not separated by tokenizer
            fields = toks[0].split('/')
            frag_parts = []
            for w in fields:
                (has_map, frag_str) = retrieve_map(w, tok_map)
                if not has_map:
                    frag_str = '(. :%s )' % w
                frag_parts.append(frag_str)
            frag_s = '(. :s/slash'
            for (i, frag_str) in enumerate(frag_parts):
                frag_s += ' :op%d %s' % ((i+1), frag_str)
            frag_s += ')'
            return (True, '[A1-1]', frag_s)
    return (False, None, None)


def get_context(toks, lemmas, poss, start, end):
    contexts = []
    n_toks = len(toks)

    prev_tok = 'SOS' if start < 1 else toks[start-1]
    prev_2tok = 'SOS' if start < 2 else toks[start-2]
    contexts.append(prev_2tok)
    contexts.append(prev_tok)

    next_tok = 'EOS' if end >= n_toks else toks[end]
    next_2tok = 'EOS' if end >= n_toks - 1 else toks[end+1]
    contexts.append(next_tok)
    contexts.append(next_2tok)

    prev_tok = 'SOS' if start < 1 else lemmas[start-1]
    prev_2tok = 'SOS' if start < 2 else lemmas[start-2]
    contexts.append(prev_2tok)
    contexts.append(prev_tok)

    next_tok = 'EOS' if end >= n_toks else lemmas[end]
    next_2tok = 'EOS' if end >= n_toks - 1 else lemmas[end+1]
    contexts.append(next_tok)
    contexts.append(next_2tok)

    prev_tok = 'SOS' if start < 1 else poss[start-1]
    prev_2tok = 'SOS' if start < 2 else poss[start-2]
    contexts.append(prev_2tok)
    contexts.append(prev_tok)

    next_tok = 'EOS' if end >= n_toks else poss[end]
    next_2tok = 'EOS' if end >= n_toks - 1 else poss[end+1]
    contexts.append(next_tok)
    contexts.append(next_2tok)
    return contexts

def extract_aligned_rules(args):
    tok_seqs = read_toks(args.tok_file)
    lemma_seqs = read_toks(args.lemma_file)
    pos_seqs = read_toks(args.pos_file)

    print 'A total of %d sentences' % len(tok_seqs)

    assert len(tok_seqs) == len(pos_seqs)
    assert len(tok_seqs) == len(lemma_seqs)

    (word2predicate, ent2frag, label2frag) = load_mappings(args.stats_dir)

    result_f = open(args.output, 'w')

    with open(args.input, 'r') as input_f:
        sent_no = 0

        while True:
            sent_map = read_sentence(input_f)
            if not sent_map:
                break

            aligned_toks = set()
            unaligned_toks = set()

            toks = tok_seqs[sent_no]
            lemmas = lemma_seqs[sent_no]
            pos_seq = pos_seqs[sent_no]
            sent_no += 1

            aligned_rules = []
            for align in sent_map:
                frag_label = align[0]
                start = align[1]
                end = align[2]
                is_ent = 'ent+' in frag_label

                is_pred = 'ARG' in frag_label

                if frag_label == 'UNKNOWN':
                    assert end - start == 1
                    unaligned_toks.add(start)
                    continue
                else:
                    new_aligned = set(xrange(start, end))
                    aligned_toks |= new_aligned
                    if is_ent: #An entity is found
                        entity_str = '_'.join(toks[start:end])
                        if entity_str in ent2frag:
                            curr_frag = ent2frag[entity_str]
                        else:
                            frag_str = build_one_entity(toks[start:end], frag_label)
                            #rule_str = '[A1-1] ## %s ## %s' % (' '.join(toks[start:end]), frag_str)
                            rule_str = '%d-%d####[A1-1] ## %s ## %s' % (start, end, ' '.join(lemmas[start:end]), frag_str)
                            aligned_rules.append(rule_str)
                            continue

                    elif is_pred:
                        assert end -start == 1
                        curr_tok = toks[start]
                        curr_lem = lemmas[start]
                        curr_pred = None
                        if curr_tok in word2predicate:
                            curr_pred = word2predicate[curr_tok]
                        elif curr_lem in word2predicate:
                            curr_pred = word2predicate[curr_lem]
                        else:
                            #curr_pred = ret_word2predicate[curr_tok]
                            curr_pred = 'UNKNOWN-01'

                        (frag_str, index, suffix) = build_one_predicate(curr_pred, frag_label)
                        #rule_str = '[A%d-%s] ## %s ## %s' % (index, suffix, curr_tok, frag_str)
                        rule_str = '%d-%d####[A%d-%s] ## %s ## %s' % (start, start+1, index, suffix, curr_lem, frag_str)
                        aligned_rules.append(rule_str)
                        continue

                    else:
                        #if frag_label in label2frag:
                        if frag_label not in label2frag:
                            print 'weird here'
                            print frag_label
                            continue

                        curr_frag = label2frag[frag_label]
                        #else:
                        #    curr_frag = ret_word2frag[frag_label]

                    new_node = FragmentHGNode(FRAGMENT_NT, -1, -1, curr_frag)
                    s = Sample(hypergraph.Hypergraph(new_node), 0)
                    new_node.cut = 1
                    new_rule, _ = s.extract_one_rule(new_node, None, curr_frag.ext_list, False)
                    rule_str = filter_vars(str(new_rule)).replace('|||', '##')
                    fields = rule_str.split('##')
                    #fields[1] = ' %s ' % ' '.join(toks[start:end])
                    fields[1] = ' %s ' % ' '.join(lemmas[start:end])
                    rule_str = '##'.join(fields)
                    rule_str = '%d-%d####%s' % (start, end, rule_str)
                    aligned_rules.append(rule_str)

            #print >>result_f, '%s ||| %s ||| %s' % (' '.join(toks), ' '.join([str(k) for k in unaligned_toks]), '++'.join(aligned_rules))
            print >>result_f, '%s ||| %s ||| %s' % (' '.join(lemmas), ' '.join([str(k) for k in unaligned_toks]), '++'.join(aligned_rules))
            assert len(aligned_toks & unaligned_toks) == 0, str(aligned_toks)+ str(unaligned_toks)
            assert len(aligned_toks | unaligned_toks) == len(toks)

        input_f.close()

    result_f.close()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--stats_dir", type=str, help="the statistics directory")
    argparser.add_argument("--input", type=str, help="the input file", required=False)
    argparser.add_argument("--output", type=str, help="the outpu file", required=False)
    #argparser.add_argument("--refine", action="store_true", help="if to refine the nonterminals")
    argparser.add_argument("--tok_file", type=str, help="the token file", required=False)
    argparser.add_argument("--lemma_file", type=str, help="the lemma file", required=False)
    argparser.add_argument("--pos_file", type=str, help="the pos tag file", required=False)
    argparser.add_argument("--ner_file", type=str, help="the ner file", required=False)
    argparser.add_argument("--tok_map", type=str, help="the tok map file", required=False)
    argparser.add_argument("--lemma_map", type=str, help="the lemma map file", required=False)
    args = argparser.parse_args()
    #extract_aligned_rules(args)
    concept_id(args)
