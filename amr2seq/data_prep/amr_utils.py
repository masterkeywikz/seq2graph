#!/usr/bin/python
import cPickle
import sys
import os
import amr_graph
from amr_graph import *
from re_utils import *

def get_amr_line(input_f):
    """Read the amr file. AMRs are separated by a blank line."""
    cur_amr=[]
    has_content=False
    for line in input_f:
      if line[0]=="(" and len(cur_amr)!=0:
         cur_amr=[]
      if line.strip()=="":
         if not has_content:
            continue
         else:
            break
      elif line.strip().startswith("#"):
        # omit the comment in the AMR file
        continue
      else:
         has_content=True
         cur_amr.append(delete_pattern(line.strip(), '~e\.[0-9]+(,[0-9]+)*'))
         #cur_amr.append(line.strip())
    return "".join(cur_amr)

#Load a list of amr graph objects
def load_amr_graphs(amr_file):

    f = open(amr_file, 'r')
    amr_line = get_amr_line(f)
    graphs = []

    curr_index = 0
    while amr_line and amr_line != '':
        #print amr_line
        #fflush(stdout)
        amr_graph = AMRGraph(amr_line)
        #print str(amr_graph)
        graphs.append(amr_graph)
        #if len(graphs) % 5000 == 0:
        #    curr_dump_file = os.path.join(divide_dir, 'graph_%d' % curr_index)
        #    curr_f = open(curr_dump_file, 'wb')
        #    cPickle.dump(graphs, curr_f)
        #    curr_f.close()
        #    curr_index += 1
        #    graphs[:] = []
        amr_line = get_amr_line(f)

    return graphs

