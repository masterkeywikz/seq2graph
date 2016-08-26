#!/usr/bin/python
import re, sys
def entities_inline(line):
    pattern = re.compile('\[([^\[\]]+)\]')
    entities = []
    line = line.strip()
    if line:
        position = 0
        #print >>wf, 'sentence %d:' % (i+1)

        while position < len(line):
            match = pattern.search(line, position)
            if not match:
                break

            match_s = match.group(1)
            match_role = match_s.split()[0]
            match_s = '_'.join(match_s.split()[1:])
            parts = match_s.split('-')

            match_s = parts[0]
            for s in parts[1:]:
                match_s += '_@-@_%s' % s
            entities.append((match_role, match_s))
            #print >>wf, match_s
            position = match.end()
    return entities

def extract_all_entities(input, output):
    pattern = re.compile('\[([^\[\]]+)\]')
    with open(input, 'r') as f:
        with open(output, 'w') as wf:
            for (i, line) in enumerate(f):
                line = line.strip()
                if line:
                    position = 0
                    print >>wf, 'sentence %d:' % (i+1)

                    while position < len(line):
                        match = pattern.search(line, position)
                        if not match:
                            break

                        match_s = match.group(1)
                        print >>wf, match_s
                        position = match.end()
                    print >>wf, ''
            wf.close()
            f.close()

if __name__ == '__main__':
    extract_all_entities(sys.argv[1], sys.argv[2])
