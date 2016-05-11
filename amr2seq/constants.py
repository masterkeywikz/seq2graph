# all the constants 
import numpy as np
import re
from os import listdir
from collections import defaultdict

RET='-RET-'
TOP='-TOP-' # top relation for the overall bracket
LBR='('
RBR=')'
SURF='-SURF-' # surface form
CONST='-CONST-'
VERB='-VERB-'
END='-END-'
UNK='UNK' # unknown words

WORD2VEC_EMBEDDING_PATH='/home/j/llc/cwang24/Tools/word2vec/GoogleNews-vectors-negative300.bin'

# DEFAULT_RULE_FILE = './rules/dep2amrLabelRules'

# def _load_rules(rule_file):
#     rf = open(rule_file,'r')
#     d = {}
#     for line in rf.readlines():
#         if line.strip():
#             dep_rel,amr_rel,_ = line.split()
#             if dep_rel not in d: d[dep_rel] = amr_rel[1:]
#         else:
#             pass
#     return d

# __DEP_AMR_REL_TABLE = _load_rules(DEFAULT_RULE_FILE)
# def get_fake_amr_relation_mapping(dep_rel):
#     return __DEP_AMR_REL_TABLE[dep_rel]

# DEFAULT_NOM_FILE = './resources/nombank-dict.1.0'

# def _read_nom_list(nombank_dict_file):
#     nomdict = open(nombank_dict_file,'r')
#     nomlist = []
#     token_re = re.compile('^\\(PBNOUN :ORTH \\"([^\s]+)\\" :ROLE-SETS')
#     for line in nomdict.readlines():
#         m = token_re.match(line.rstrip())
#         if m:
#             nomlist.append(m.group(1))
#     return nomlist

# NOMLIST = _read_nom_list(DEFAULT_NOM_FILE)

# DEFAULT_BROWN_CLUSTER = './resources/wclusters-engiga'
    
# def _load_brown_cluster(dir_path,cluster_num=1000):
#     cluster_dict = defaultdict(str)
#     for fn in listdir(dir_path):
#         if re.match('^.*c(\d+).*$',fn).group(1) == str(cluster_num) and fn.endswith('.txt'):
#             with open(dir_path+'/'+fn,'r') as f:
#                 for line in f:
#                     bitstring, tok, freq = line.split()
#                     cluster_dict[tok]=bitstring

#     return cluster_dict

# BROWN_CLUSTER=_load_brown_cluster(DEFAULT_BROWN_CLUSTER)

PATH_TO_VERB_LIST = './resources/verbalization-list-v1.01.txt'

def _load_verb_list(path_to_file):
    verbdict = {}
    with open(path_to_file,'r') as f:
        for line in f:
            if not line.startswith('#') and line.strip():
                if not line.startswith('DO-NOT-VERBALIZE'):
                    verb_type, lemma, _, subgraph_str = re.split('\s+',line,3)
                    subgraph = {}
                
                    #if len(l) == 1: 
                    #else: # have sub-structure
                    root = re.split('\s+', subgraph_str, 1)[0]
                    subgraph[root] = {}
                    for match in re.finditer(':([^\s]+)\s*([^\s:]+)',subgraph_str):
                        relation = match.group(1)
                        concept = match.group(2)
                        subgraph[root][relation] = concept
                        
                    verbdict[lemma] = verbdict.get(lemma,[])
                    verbdict[lemma].append(subgraph)

    return verbdict

VERB_LIST = _load_verb_list(PATH_TO_VERB_LIST)

# PATH_TO_COUNTRY_LIST='./resources/country-list.csv'

# def _load_country_list(path_to_file):
#     countrydict = {}
#     with open(path_to_file,'r') as f:
#         for line in f:
#             line = line.strip()
#             country_name, country_adj, _ = line.split(',', 2)
#             countrydict[country_adj] = country_name

#     return countrydict
    
# COUNTRY_LIST=_load_country_list(PATH_TO_COUNTRY_LIST)
                

# given different domain, return range of split corpus #TODO: move this part to config file
def get_corpus_range(corpus_section,corpus_type):
    DOMAIN_RANGE_TABLE={ \
        'train':{
            'proxy':(0,6603),
            'bolt':(6603,7664),
            'dfa':(7664,9367),
            'mt09sdf':(9367,9571),
            'xinhua':(9571,10312)
        },
        'dev':{
            'proxy':(0,826),
            'bolt':(826,959),
            'consensus':(959,1059),
            'dfa':(1059,1269),
            'xinhua':(1269,1368)
        },
        'test':{
            'proxy':(0,823),
            'bolt':(823,956),
            'consensus':(956,1056),
            'dfa':(1056,1285),
            'xinhua':(1285,1371)
        }
    }

    return DOMAIN_RANGE_TABLE[corpus_type][corpus_section]