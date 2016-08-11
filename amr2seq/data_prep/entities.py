#!/usr/bin/python
import re, sys, os
import cPickle
from identify_entity import entities_inline
def identify_entities(tok_file, ner_file, entity_set):
    all_entities = []
    with open(tok_file, 'r') as tok_f:
        with open(ner_file, 'r') as ner_f:
            for (i, tok_line) in enumerate(tok_f):
                sent_entities = []
                ner_line = ner_f.readline()
                tok_line = tok_line.strip()
                #print 'sentence %d:' % (i+1)
                if tok_line:
                    aligned_toks = set()

                    toks = tok_line.split()
                    matched_entities = entities_inline(ner_line)
                    entities = {}
                    for (role, ent) in matched_entities:
                        entities[ent] = role
                    #entities = set(entities_inline(ner_line))

                    length = len(toks)
                    for start in xrange(length):
                        if start in aligned_toks:
                            continue

                        for span in xrange(length+1, 0, -1):
                            end = start + span
                            if end-1 in aligned_toks:
                                continue

                            if end > length:
                                continue

                            curr_str = '_'.join(toks[start:end])
                            if curr_str in entities:
                                curr_set = set(xrange(start, end))
                                aligned_toks |= curr_set
                                sent_entities.append((start, end, entities[curr_str]))

                                #print '%d-%d : ' % (start, end), curr_str

                            elif curr_str in entity_set:
                                curr_set = set(xrange(start, end))
                                aligned_toks |= curr_set
                                sent_entities.append((start, end, None))
                                #print '%d-%d : ' % (start, end), curr_str
                all_entities.append(sent_entities)
    return all_entities

def load_entities(stats_dir):
    stats_files = os.popen('ls %s/stat_*' % stats_dir).read().split()
    all_entities = set()

    all_non_pred = set()
    all_pred = set()

    assert len(stats_files) == 8
    for file in stats_files:
        f = open(file, 'rb')
        stats = cPickle.load(f)
        entity_words = stats[3]

        all_pred |= stats[0]
        all_non_pred |= set(stats[6].keys())

        all_entities |= entity_words
        f.close()

    all_entities -= all_non_pred
    all_entities -= all_pred
    #for entity in all_entities:
    #    if entity in all_non_pred:
    #        continue
    #    if entity in all_pred:
    #        continue
    #    #print entity

    return all_entities

if __name__ == '__main__':
    all_entities = load_entities(sys.argv[3])
    identify_entities(sys.argv[1], sys.argv[2], all_entities)
