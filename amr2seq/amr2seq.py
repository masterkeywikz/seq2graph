'''
categorize amr; generate linearized amr sequence
'''
import sys, os, re, codecs
import string
import gflags
from amr_graph import AMR
from collections import OrderedDict, defaultdict
#from constants import TOP,LBR,RBR,RET,SURF,CONST,END,VERB
from constants import *
from parser import ParserError
from date_extraction import *
FLAGS=gflags.FLAGS
gflags.DEFINE_string("version",'1.0','version for the sequence generated')
gflags.DEFINE_integer("min_prd_freq",50,"threshold for filtering out predicates")
gflags.DEFINE_integer("min_var_freq",50,"threshold for filtering out non predicate variables")
gflags.DEFINE_string("seq_cat",'freq',"mode to output sequence category")
class AMR_stats(object):
    def __init__(self):
        self.num_reentrancy = 0
        self.num_predicates = defaultdict(int)
        self.num_nonpredicate_vals = defaultdict(int)
        self.num_consts = defaultdict(int)
        self.num_entities = defaultdict(int)
        self.num_named_entities = defaultdict(int)

    def collect_stats(self, amrs):
        for amr in amrs:
            named_entity_nums, entity_nums, predicate_nums, variable_nums, const_nums, reentrancy_nums = amr.statistics()

            self.update(reentrancy_nums, predicate_nums, variable_nums, const_nums, entity_nums, named_entity_nums)

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

        dump_file(pred_f, self.num_predicates)
        dump_file(non_pred_f, self.num_nonpredicate_vals)
        dump_file(const_f, self.num_consts)
        dump_file(entity_f, self.num_entities)
        dump_file(named_entity_f, self.num_named_entities)

    def __str__(self):
        s = ''
        s += 'Total number of reentrancies: %d\n' % self.num_reentrancy
        s += 'Total number of predicates: %d\n' % len(self.num_predicates)
        s += 'Total number of non predicates variables: %d\n' % len(self.num_nonpredicate_vals)
        s += 'Total number of constants: %d\n' % len(self.num_consts)
        s += 'Total number of entities: %d\n' % len(self.num_entities)
        s += 'Total number of named entities: %d\n' % len(self.num_named_entities)
        return s


class AMR_seq:
    def __init__(self, stats=None):
        self.stats = stats
        self.min_prd_freq = FLAGS.min_prd_freq
        self.min_var_freq = FLAGS.min_var_freq
        #pass

    def linearize_amr(self, instance):
        '''
        given an amr graph, output a linearized and categorized amr sequence;
        TODO: use dfs prototype
        '''
        #pass

        amr = instance[0]

        r = amr.roots[0] # single root assumption
        old_depth = -1
        depth = -1
        stack = [(r,TOP,None,0)]
        aux_stack = []
        seq = []

        while stack:
            old_depth = depth
            cur_var, rel, parent, depth = stack.pop()
            #exclude_rels = []

            i = 0
            #print seq
            #print stack
            #print aux_stack
            while old_depth - depth >= i:
                if aux_stack == []:
                    import pdb
                    pdb.set_trace()
                seq.append(aux_stack.pop())
                i+=1


            if (parent, rel, cur_var) in amr.reentrance_triples:
                #seq.extend([rel+LBR,RET,RBR+rel])
                seq.append(rel+LBR)
                seq.append(RET)
                aux_stack.append(RBR+rel)
                continue

            seq.append(rel+LBR)
            exclude_rels, cur_symbol = self.get_symbol(cur_var, instance, mode=FLAGS.seq_cat)
            seq.append(cur_symbol)
            aux_stack.append(RBR+rel)

            for rel, var in reversed(amr[cur_var].items()):
                if rel not in exclude_rels:
                    stack.append((var[0],rel,cur_var,depth+1))

        seq.extend(aux_stack[::-1])
        #seq.append(END)

        return seq

    def _get_pred_symbol(self, var, instance, mode):
        amr, alignment, tok, pos = instance
        if mode == 'basic':
            pred_name = amr.node_to_concepts[var]
            return pred_name
        elif mode == 'freq':
            pred_name = amr.node_to_concepts[var]
            if self.stats.num_predicates[pred_name] >= self.min_prd_freq:
                return pred_name
            else:
                sense = pred_name.split('-')[-1]
                return VERB+sense

                # TODO further categorize the verb type using verbalization list
        else:
            raise Exception('Unexpected mode %s' % (mode))

    def _get_variable_symbal(self, var, instance, mode):
        amr, alignment, tok, pos = instance
        if mode == 'basic':
            variable_name = amr.node_to_concepts[var]
            return variable_name
        elif mode == 'freq':
            variable_name = amr.node_to_concepts[var]
            if self.stats.num_nonpredicate_vals[variable_name] >= self.min_var_freq:
                return variable_name
            else:
                return SURF

                # TODO further categorize the variable type
        else:
            raise Exception('Unexpected mode %s' % (mode))

    def get_symbol(self, var, instance, mode='basic'):
        '''
        get symbol for each amr concept and variable
        mode:
            "basic": no frequence filtering
            "freq": frequence filtering
        '''
        amr = instance[0]

        if amr.is_named_entity(var):
            exclude_rels = ['wiki','name']
            entity_name = amr.node_to_concepts[var]
            return exclude_rels, 'NE_'+entity_name
        elif amr.is_entity(var):
            entity_name = amr.node_to_concepts[var]
            return [], 'ENT_'+entity_name
        elif amr.is_predicate(var):
            return [], self._get_pred_symbol(var, instance, mode=mode)
        elif amr.is_const(var):
            if var in ['interrogative', 'imperative', 'expressive', '-']:
                return [], var
            else:
                return [], CONST

        else:
            #variable_name = amr.node_to_concepts[var]
            #return [], variable_name
            return [], self._get_variable_symbal(var, instance, mode=mode)

        return [],var

    def restore_amr(self, tok_seq, lemma_seq, amrseq, repr_map):
        '''
        Given a linearized amr sequence, restore its amr graph
        Deal with non-matching parenthesis
        '''
        def rebuild_seq(parsed_seq):
            new_seq = []
            stack = []
            try:

                while parsed_seq[-1][1] == "LPAR": #Delete the last few left parenthesis
                    parsed_seq = parsed_seq[:-1]

                assert len(parsed_seq) > 0, parsed_seq
                for (token, type) in parsed_seq:
                    if type == "LPAR": #Left parenthesis
                        if stack and stack[-1][1] == "LPAR":
                            new_token = 'ROOT'
                            new_type = 'NONPRED'
                            stack.append((new_token, new_type))
                            new_seq.append((new_token, new_type))

                        stack.append((token, type))
                        new_seq.append((token, type))
                    elif type == "RPAR": #Right parenthesis
                        assert stack
                        if stack[-1][1] == "LPAR": #No concept for this edge, remove
                            stack.pop()
                            new_seq.pop()
                        elif stack[-2][0][:-1] == '-TOP-':
                            continue
                        else:
                            stack.pop()
                            ledgelabel, ltype = stack.pop()
                            try:
                                assert ltype == "LPAR", ('%s %s'% (ledgelabel, ltype))
                            except:
                                print stack
                                print ledgelabel, ltype
                                print token, type
                                sys.exit(1)
                            redgelabel = ')%s' % (ledgelabel[:-1])
                            new_seq.append((redgelabel, "RPAR"))
                    else:
                        if stack[-1][1] == "LPAR":
                            stack.append((token, type))
                            new_seq.append((token, type))
                while stack:
                    while stack[-1][1] == "LPAR" and stack:
                        stack.pop()

                    if not stack:
                        break

                    stack.pop()
                    ledgelabel, ltype = stack.pop()
                    assert ltype == "LPAR"
                    redgelabel = ')%s' % (ledgelabel[:-1])
                    new_seq.append((redgelabel, "RPAR"))

                return new_seq
            except:
                print parsed_seq
                print new_seq
                sys.exit(1)

        def make_compiled_regex(rules):
            regexstr =  '|'.join('(?P<%s>%s)' % (name, rule) for name, rule in rules)
            return re.compile(regexstr)

        def register_var(token):
            num = 0
            while True:
                currval = ('%s%d' % (token[0], num)) if token[0] in string.letters else ('a%d' % num)
                if currval in var_set:
                    num += 1
                else:
                    var_set.add(currval)
                    return currval

        def buildLinearEnt(entity_name, ops, wiki_label):
            ops_strs = ['op%d( "%s" )op%d' % (index, s, index) for (index, s) in enumerate(ops, 1)]
            new_op_strs = []
            for tok in ops_strs:
                if '"("' in tok or '")"' in tok:
                    break
                new_op_strs.append(tok)
            wiki_str = 'wiki( "%s" )wiki' % wiki_label
            ent_repr = '%s %s name( name %s )name' % (entity_name, wiki_str, ' '.join(new_op_strs))
            #ent_repr = '%s %s name( name %s )name' % (entity_name, wiki_str, ' '.join(ops_strs))
            return ent_repr

        def buildLinearVerbal(lemma, node_repr):
            assert lemma in VERB_LIST
            subgraph = None
            for g in VERB_LIST[lemma]:
                if node_repr in g:
                    subgraph = g
                    break

            verbal_repr = node_repr
            for rel, subj in subgraph[node_repr].items():
                verbal_repr = '%s %s( %s )%s' % (verbal_repr, rel, subj, rel)
            return verbal_repr

        def buildLinearDate(rels):
            date_repr = 'date-entity'
            for rel, subj in rels:
                date_repr = '%s %s( %s )%s' % (date_repr, rel, subj, rel)
            return date_repr

        amr = AMR()
        #print amrseq.strip()
        seq = amrseq.strip().split()

        new_seq = []

        #Deal with redundant RET
        skip = False
        for (i, tok) in enumerate(seq):
            if skip:
                skip = False
                continue
            if tok in repr_map:
                #tok_nosuffix = re.sub('-[0-9]+', '', tok)

                start, end, node_repr, wiki_label = repr_map[tok]

                if 'NE' in tok: #Is a named entity
                    #print ' '.join(tok_seq[start:end])
                    branch_form = buildLinearEnt(node_repr, tok_seq[start:end], wiki_label)  #Here rebuild the op representation of the named entity
                    new_seq.append(branch_form)
                elif 'VERBAL' in tok:  #Is in verbalization list, should rebuild
                    assert end == start + 1
                    branch_form = buildLinearVerbal(lemma_seq[start], node_repr)
                    new_seq.append(branch_form)
                elif 'DATE' in tok: #Rebuild a date entity
                    rels = dateRepr(tok_seq[start:end])
                    branch_form = buildLinearDate(rels)
                    new_seq.append(branch_form)
                else:
                    new_seq.append(node_repr)
            else:
                if 'NE' in tok:
                    tok = re.sub('-[0-9]+', '', tok)
                    tok = tok[3:]
                elif 'ENT' in tok:
                    tok = re.sub('-[0-9]+', '', tok)
                    tok = tok[4:]
                elif 'RET' in tok or isSpecial(tok):  #Reentrancy, currently not supported
                    prev_elabel = new_seq[-1]
                    assert prev_elabel[-1] == '(', prev_elabel
                    prev_elabel = prev_elabel[:-1]
                    if i +1 < len(seq):
                        next_elabel = seq[i+1]
                        if next_elabel[0] == ')':
                            next_elabel = next_elabel[1:]
                            if prev_elabel == next_elabel:
                                skip = True
                                new_seq.pop()
                                continue
                    else:
                        new_seq.pop()
                        continue
                        #print 'Weird here'
                        #print seq

                new_seq.append(tok)

        amrseq = ' '.join(new_seq)
        seq = amrseq.split()
        triples = []

        stack = []
        state = 0
        node_idx = 0; # sequential new node index
        mapping_table = {};  # old new index mapping table

        var_set = set()

        const_set = set(['interrogative', 'imperative', 'expressive', '-'])
        lex_rules = [
            ("LPAR", '[^\s()]+\('),  #Start of an edge
            ("RPAR",'\)[^\s()]+'),  #End of an edge
            #("SURF", '-SURF-'),  #Surface form non predicate
            ("VERB", '-VERB-\d+'), # predicate
            ("CONST", '"[^"]+"'), # const
            ("REENTRANCY", '-RET-'),  #Reentrancy
            ("ENTITY", 'ENT_([^\s()]+)'),  #Entity
            ("NER", 'NE_([^\s()]+)'), #Named entity
            ("PRED", '([^\s()]+)-[0-9]+'), #Predicate
            ("NONPRED", '([^\s()]+)'),  #Non predicate variables
            ("POLARITY", '\s(\-|\+)(?=[\s\)])')
        ]

        token_re = make_compiled_regex(lex_rules)

        parsed_seq = []
        for match in token_re.finditer(amrseq):
            token = match.group()
            type = match.lastgroup
            parsed_seq.append((token, type))

        PNODE = 1
        CNODE = 2
        LEDGE = 3
        REDGE = 4
        RCNODE = 5
        parsed_seq = rebuild_seq(parsed_seq)

        token_seq = [token for (token, type) in parsed_seq]

        seq_length = len(parsed_seq)
        for (currpos, (token, type)) in enumerate(parsed_seq):
            if state == 0: #Start state
                assert type == "LPAR", ('start with symbol: %s' % token)
                edgelabel = token[:-1]
                stack.append((LEDGE, edgelabel))
                state = 1

            elif state == 1: #Have just identified an left edge, next expect a concept
                if type == "NER":
                    nodelabel = register_var(token)
                    nodeconcept = token
                    stack.append((PNODE,nodelabel,nodeconcept))
                    state = 2
                elif type == "ENTITY":
                    nodelabel = register_var(token)
                    nodeconcept = token
                    stack.append((PNODE,nodelabel,nodeconcept))
                    state = 2
                elif type == "PRED":
                    nodelabel = register_var(token)
                    nodeconcept = token
                    stack.append((PNODE,nodelabel,nodeconcept))
                    state = 2
                elif type == "NONPRED":
                    nodelabel = register_var(token)
                    nodeconcept = token
                    stack.append((PNODE,nodelabel,nodeconcept))
                    state = 2
                elif type == "CONST":
                    #if currpos + 1 < seq_length and parsed_seq[currpos+1][1] == "LPAR":
                    #    nodelabel = register_var(token)
                    #    nodeconcept = token
                    #    stack.append((PNODE,nodelabel,nodeconcept))
                    #else:
                    stack.append((PNODE,token.strip(),None))
                    state = 2
                elif type == "REENTRANCY":
                    if currpos + 1 < seq_length and parsed_seq[currpos+1][1] == "LPAR":
                        nodelabel = register_var(token)
                        nodeconcept = token
                        stack.append((PNODE,nodelabel,nodeconcept))
                    else:
                        stack.append((PNODE,token.strip(),None))
                    state = 2
                elif type == "POLARITY":
                    if currpos + 1 < seq_length and parsed_seq[currpos+1][1] == "LPAR":
                        nodelabel = register_var(token)
                        nodeconcept = token
                        stack.append((PNODE,nodelabel,nodeconcept))
                    else:
                        stack.append((PNODE,token.strip(),None))
                    state = 2
                else: raise ParserError , "Unexpected token %s"%(token.encode('utf8'))

            elif state == 2: #Have just identified a PNODE concept
                if type == "LPAR":
                    edgelabel = token[:-1]
                    stack.append((LEDGE, edgelabel))
                    state = 1
                elif type == "RPAR":
                    assert stack[-1][0] == PNODE
                    forgetme, nodelabel, nodeconcept = stack.pop()
                    if not nodelabel in amr.node_to_concepts and nodeconcept is not None:
                        amr.node_to_concepts[nodelabel] = nodeconcept

                    foo = amr[nodelabel]
                    if stack and stack[-1][1] != "-TOP-": #This block needs to be updated
                        stack.append((CNODE, nodelabel, nodeconcept))
                        state = 3
                    else: #Single concept AMR
                        assert len(stack) == 1 and stack[-1][1] == "-TOP-", "Not start with TOP"
                        stack.pop()
                        if amr.roots:
                            break
                        amr.roots.append(nodelabel)
                        state = 0
                        #break
                else: raise ParserError, "Unexpected token %s"%(token)

            elif state == 3: #Have just finished a CNODE, which means wrapped up with one branch
                if type == "LPAR":
                    edgelabel = token[:-1]
                    stack.append((LEDGE, edgelabel))
                    state = 1

                elif type == "RPAR":
                    edges = []
                    while stack[-1][0] != PNODE:
                        children = []
                        assert stack[-1][0] == CNODE, "Expect a parsed node but none found"
                        forgetme, childnodelabel, childconcept = stack.pop()
                        children.append((childnodelabel,childconcept))

                        assert stack[-1][0] == LEDGE, "Found a non-left edge"
                        forgetme, edgelabel = stack.pop()

                        edges.append((edgelabel,children))

                    forgetme,parentnodelabel,parentconcept = stack.pop()

                    #check for annotation error
                    if parentnodelabel in amr.node_to_concepts:
                        print parentnodelabel, parentconcept
                        assert parentconcept is not None
                        if amr.node_to_concepts[parentnodelabel] == parentconcept:
                            sys.stderr.write("Wrong annotation format: Revisited concepts %s should be ignored.\n" % parentconcept)
                        else:
                            sys.stderr.write("Wrong annotation format: Different concepts %s and %s have same node label(index)\n" % (amr.node_to_concepts[parentnodelabel],parentconcept))
                            parentnodelabel = parentnodelabel + "1"


                    if not parentnodelabel in amr.node_to_concepts and parentconcept is not None:
                        amr.node_to_concepts[parentnodelabel] = parentconcept

                    for edgelabel,children in reversed(edges):
                        hypertarget = []
                        for node, concept in children:
                            if node is not None and not node in amr.node_to_concepts and concept:
                                amr.node_to_concepts[node] = concept
                            hypertarget.append(node)
                        hyperchild = tuple(hypertarget)
                        amr._add_triple(parentnodelabel,edgelabel,hyperchild)

                    if stack and stack[-1][1] != "-TOP-": #we have done with current level
                        state = 3
                        stack.append((CNODE, parentnodelabel, parentconcept))
                    else: #Single concept AMR
                        assert len(stack) == 1 and stack[-1][1] == "-TOP-", "Not start with TOP"
                        stack.pop()
                        if amr.roots:
                            break
                        amr.roots.append(parentnodelabel)
                        state = 0
                        break
                        #state = 0
                        #amr.roots.append(parentnodelabel)
                else: raise ParserError, "Unexpected token %s"%(token.encode('utf8'))

        if state != 0 and stack:
            raise ParserError, "mismatched parenthesis"
        return amr




def readAMR(amrfile_path):
    amrfile = codecs.open(amrfile_path,'r',encoding='utf-8')
    #amrfile = open(amrfile_path,'r')
    comment_list = []
    comment = OrderedDict()
    amr_list = []
    amr_string = ''

    for line in amrfile.readlines():
        if line.startswith('#'):
            for m in re.finditer("::([^:\s]+)\s(((?!::).)*)",line):
                #print m.group(1),m.group(2)
                comment[m.group(1)] = m.group(2)
        elif not line.strip():
            if amr_string and comment:
                comment_list.append(comment)
                amr_list.append(amr_string)
                amr_string = ''
                comment = {}
        else:
            amr_string += line.strip()+' '

    if amr_string and comment:
        comment_list.append(comment)
        amr_list.append(amr_string)
    amrfile.close()

    return (comment_list,amr_list)

def amr2sequence(toks, amr_graphs, alignments, poss, out_seq_file, amr_stats):
    amr_seq = AMR_seq(stats=amr_stats)
    with open(out_seq_file, 'w') as outf:
        print 'Linearizing ...'
        for i,g in enumerate(amr_graphs):
            print 'No ' + str(i) + ':' + ' '.join(toks[i])
            instance = (g, alignments[i], toks[i], poss[i])
            seq = ' '.join(amr_seq.linearize_amr(instance))
            print >> outf, seq

def sequence2amr(toks, lemmas, amrseqs, cate_to_repr, out_amr_file):
    amr_seq = AMR_seq()
    with open(out_amr_file, 'w') as outf:
        print 'Restoring AMR graphs ...'
        for i, (s, repr_map) in enumerate(zip(amrseqs, cate_to_repr)):
            print 'No ' + str(i) + ':' + ' '.join(toks[i])
            #print repr_map
            restored_amr = amr_seq.restore_amr(toks[i], lemmas[i], s, repr_map)
            if 'NONE' in restored_amr.to_amr_string():
                print s
                print repr_map
            print >> outf, restored_amr.to_amr_string()
            print >> outf, ''
        outf.close()

def isSpecial(symbol):
    for l in ['ENT', 'NE', 'VERB', 'SURF', 'CONST']:
        if l in symbol:
            return True
    return False

if __name__ == "__main__":

    gflags.DEFINE_string("train_data_dir",'../train',"train data directory")
    gflags.DEFINE_string("data_dir",'../train',"data directory")
    gflags.DEFINE_string("amrseq_file",'../dev.decode.amrseq',"amr sequence file")

    gflags.DEFINE_boolean("seq2amr", False, "If sequence to amr")
    gflags.DEFINE_boolean("amr2seq", False, "If amr to sequence")
    argv = FLAGS(sys.argv)

    amr_file = os.path.join(FLAGS.data_dir, 'amr')

    alignment_file = os.path.join(FLAGS.data_dir, 'alignment')
    sent_file = os.path.join(FLAGS.data_dir, 'sentence')
    tok_file = os.path.join(FLAGS.data_dir, 'token')
    lemma_file = os.path.join(FLAGS.data_dir, 'lemmatized_token')
    pos_file = os.path.join(FLAGS.data_dir, 'pos')
    map_file = os.path.join(FLAGS.data_dir, 'cate_map')
    cate_to_repr = []   #Maintain a map for each sentence

    comment_list, amr_list = readAMR(amr_file)
    if FLAGS.seq2amr:
        amrseqs = [line.strip() for line in open(FLAGS.amrseq_file, 'r')]
        for line in open(map_file):
            curr_map = {}
            if line.strip():
                fields = line.strip().split('##')
                for map_tok in fields:
                    fs = map_tok.strip().split('++')
                    try:
                        assert len(fs) == 5, line
                    except:
                        print line
                        print fields
                        print curr_map
                        print fs
                        sys.exit(1)
                    start = int(fs[1])
                    end = int(fs[2])
                    curr_map[fs[0]] = (start, end, fs[3], fs[4])
            cate_to_repr.append(curr_map)
        assert len(cate_to_repr) == len(amrseqs)

    sents = [line.strip().split() for line in open(sent_file, 'r')]
    toks = [line.strip().split() for line in open(tok_file, 'r')]
    lemmas = [line.strip().split() for line in open(lemma_file, 'r')]
    poss = [line.strip().split() for line in open(pos_file, 'r')]

    if FLAGS.amr2seq:
        amr_graphs = [AMR.parse_string(amr_string) for amr_string in amr_list]
        alignments = [line.strip().split() for line in open(alignment_file, 'r')]

        ##################
        # get statistics
        ##################
        if FLAGS.train_data_dir == FLAGS.data_dir:
            train_amr_graphs = amr_graphs
        else:
            train_amr_file = os.path.join(FLAGS.train_data_dir, "amr")
            _ , train_amr_list = readAMR(train_amr_file)
            train_amr_graphs = [AMR.parse_string(amr_string) for amr_string in train_amr_list]

        amr_stats = AMR_stats()
        amr_stats.collect_stats(train_amr_graphs)
        print amr_stats

        seq_result_file = os.path.join(FLAGS.data_dir, 'amrseq.%s' % FLAGS.version)
        amr2sequence(toks, amr_graphs, alignments, poss, seq_result_file, amr_stats)

    if FLAGS.seq2amr:
        amr_result_file = os.path.join(FLAGS.data_dir, 'amr.%s' % FLAGS.version)
        sequence2amr(toks, lemmas, amrseqs, cate_to_repr, amr_result_file)


