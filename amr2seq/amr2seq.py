'''
categorize amr; generate linearized amr sequence
'''
import sys, os, re, codecs
from amr_graph import AMR
from collections import OrderedDict, defaultdict
from constants import TOP,LBR,RBR,RET,SURF,END
import gflags
FLAGS=gflags.FLAGS

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
    def __init__(self):
        pass

    def linearize_amr(self, amr):
        '''
        given an amr graph, output a linearized and categorized amr sequence;
        TODO: use dfs prototype
        '''
        #pass
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
            exclude_rels, cur_symbol = self.get_symbol(cur_var, amr)
            seq.append(cur_symbol)
            aux_stack.append(RBR+rel)

            for rel, var in reversed(amr[cur_var].items()):
                if rel not in exclude_rels:
                    stack.append((var[0],rel,cur_var,depth+1))
                    
        seq.extend(aux_stack[::-1])
        seq.append(END)

        return seq

    def get_symbol(self, var, amr):
        if amr.is_named_entity(var):
            exclude_rels = ['wiki','name']
            entity_name = amr.node_to_concepts[var]
            return exclude_rels, 'NE_'+entity_name
        elif amr.is_entity(var):
            entity_name = amr.node_to_concepts[var]
            return [], 'ENT_'+entity_name
        elif amr.is_predicate(var):
            pred_name = amr.node_to_concepts[var]
            return [], pred_name
        elif amr.is_const(var):
            if var in ['interrogative', 'imperative', 'expressive', '-']:
                return [], var
            else:
                return [], SURF

        else:
            variable_name = amr.node_to_concepts[var]
            return [], variable_name

        return [],var
        

            
        

    
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

if __name__ == "__main__":
    gflags.DEFINE_string("data_dir",'../train',"data directory")
    argv = FLAGS(sys.argv)

    amr_file = os.path.join(FLAGS.data_dir, 'amr')
    alignment_file = os.path.join(FLAGS.data_dir, 'alignment')
    sent_file = os.path.join(FLAGS.data_dir, 'sentence')
    tok_file = os.path.join(FLAGS.data_dir, 'token')
    #lemma_file = os.path.join(FLAGS.data_dir, 'lemma')
    pos_file = os.path.join(FLAGS.data_dir, 'pos')

    comment_list, amr_list = readAMR(amr_file)
    amr_graphs = [AMR.parse_string(amr_string) for amr_string in amr_list]
    alignments = [line.strip().split() for line in open(alignment_file, 'r')]
    sents = [line.strip().split() for line in open(sent_file, 'r')]
    toks = [line.strip().split() for line in open(tok_file, 'r')]
    #lemmas = [line.strip().split() for line in open(lemma_file, 'r')]
    poss = [line.strip().split() for line in open(pos_file, 'r')]

    ##################
    # get statistics 
    ##################
    # amr_stats = AMR_stats()
    # amr_stats.collect_stats(amr_graphs)
    # print amr_stats
    # amr_stats.dump2dir(FLAGS.data_dir)
    
    # print toks[1]
    # print amr_graphs[1].to_amr_string()
    # amr_seq = AMR_seq()
    # print ' '.join(amr_seq.linearize_amr(amr_graphs[1]))
    out_seq_file = os.path.join(FLAGS.data_dir, 'amrseq')
    amr_seq = AMR_seq()
    with open(out_seq_file, 'w') as outf:
        print 'Linearizing ...'
        for i,g in enumerate(amr_graphs):
            print 'No ' + str(i) + ':' + ' '.join(toks[i])
            #print amr_graphs[i].to_amr_string()
            seq = ' '.join(amr_seq.linearize_amr(g))
            #ourf.write(seq)
            print >> outf, seq
        
    
