#!/usr/bin/python

#Compute number of toks in span that also appear in the fragment
def similarity(toks, oth_toks):
    sim_score = 0
    for tok in toks:
        for oth_tok in oth_toks:
            if tok.lower() == oth_tok.lower():
                sim_score += 1
                break
    return sim_score

#Given the entity mention of the fragment
#Remove all mentions that are redundant
def removeRedundant(toks, entity_spans, op_toks):
    all_spans = []
    for (start, end) in entity_spans:
        sim_score = similarity(toks[start:end], op_toks)
        all_spans.append((sim_score, start, end))

    all_spans = sorted(all_spans, key=lambda x: -x[0])

    max_score = all_spans[0][0]
    remained_spans = [(start, end) for (sim_score, start, end) in all_spans if sim_score == max_score]
    return remained_spans

def removeDateRedundant(date_spans):
    max_span = max([end-start for (start, end) in date_spans])
    remained_spans = [(start, end) for (start, end) in date_spans if end-start == max_span]
    return remained_spans
