#!/usr/bin/python
import sys, re
from collections import defaultdict
months = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12, 'Sept.':9}
timezone = set(['UTC', 'GMT', 'AEDT', 'EST', 'DST'])
days = set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
date_suffixes = set(['th', 'st', 'nd', 's'])
day_zone = set(['morning', 'evening', 'afternoon', 'night', 'noon', 'midday', 'midnight'])
seasons = set(['spring', 'summer', 'fall', 'winter'])
def isNum(s):
    regex = re.compile('[0-9]+([^0-9\s]*)')
    match = regex.match(s)
    return match and len(match.group(1)) == 0

def monthAbbre(month_map):
    abbr_map = {}
    for m in month_map:
        abbr_m = m[:3] + '.'
        abbr_map[abbr_m] = month_map[m]
    return abbr_map

def lowerMap(orig_map):
    new_map = {}
    for tok in orig_map:
        lower_tok = tok.lower()
        new_map[lower_tok] = orig_map[tok]
    return new_map

def lowerSet(orig_set):
    new_set = set()
    for tok in orig_set:
        new_set.add(tok.lower())
    return new_set

def extractTemplate(s):
    fields = s.split()
    template = []
    #month_map = lowerMap(months)
    month_map = months
    abbr_month = monthAbbre(month_map)

    #day_map = lowerSet(days)
    day_map = days
    #abbr_day = monthAbbre(day_map)

    for tok in fields:
        if isYear(tok):
            template.append('YEAR')
        elif tok in month_map or tok in abbr_month:
            template.append('MONTH')
        elif tok in day_map:
            template.append('DAY')
        elif tok in day_zone:
            template.append('DAY_ZONE')
        elif isNum(tok):
            template.append('NUM%d' % len(tok))
        else:
            template.append(tok)
    return ' '.join(template)

def dateRepr(toks):
    rels = []

    exist_rels = set()

    month_map = months
    abbr_month = monthAbbre(month_map)

    day_map = days
    has_year = False
    has_month = False
    has_century = False

    for tok in toks:
        if isYear(tok):
            has_year = True
        elif tok in month_map:
            has_month = True
        elif tok in abbr_month:
            has_month = True
        elif tok == 'century':
            has_century = True
        #elif tok == 's':

    for tok in toks:
        if isYear(tok):
            rels.append(('year', str(int(tok))))

        elif tok in month_map:
            rels.append(('month', str(month_map[tok])))
        elif tok in abbr_month:
            rels.append(('month', str(abbr_month[tok])))
        elif tok in day_map:
            rels.append(('weekday', tok.lower()))
        elif tok in day_zone:
            rels.append(('dayperiod', tok))
        elif isNum(tok):
            if len(tok) == 8: #YYYYMMDD
                rels.append(('day', str(int(tok[6:]))))
                rels.append(('month', str(int(tok[4:6]))))
                rels.append(('year', str(int(tok[:4]))))
            elif len(tok) == 6: #YYMMDD
                year = int(tok[:2])
                if year < 50: #20YY
                    rels.append(('year', '20%d' % year))
                else:
                    rels.append(('year', '19%d' % year))
                month = int(tok[2:4])
                if month > 0 and month <= 12:
                    rels.append(('month', '%d' % month))
                day = int(tok[4:])
                if day > 0 and day <= 31:
                    rels.append(('day', '%d' % day))
            elif len(tok) == 2:
                if has_century:
                    rels.append(('century', str(int(tok))))
                elif not has_month:
                    rels.append(('month', str(int(tok))))
                    has_month = True
                else:
                    rels.append(('day', str(int(tok))))

            elif len(tok) == 1:
                if not has_month and int(tok) < 12:
                    rels.append(('month', str(int(tok))))
                else:
                    rels.append(('day', str(int(tok))))
        elif tok in seasons:
            rels.append(('season', tok))
    if not rels:
        print 'weird: %s' % (' '.join(toks))
    return rels

def loadTemplates(file):
    temp_map = {}
    for line in open(file):
        if line.strip():
            fields = line.strip().split(' #### ')
            temp_map[fields[0]] = int(fields[1])
    return temp_map

def isYear(tok):
    if isNum(tok) and len(tok) == 4:
        first_two = int(tok[:2])
        return first_two == 19 or first_two == 20
    return False

def isDay(tok):
    if not isNum(tok):
        return False
    date = int(tok)
    return date >= 0 and date <= 31

def mergeDates(dates_in_line, toks):
    unalign_symbols = set(['@:@', 'of', 'on', 'at', ',', ':'])
    prev_start = None
    prev_end = None
    new_spans = []
    for (start, end) in dates_in_line:
        if prev_end is not None:
            between_set = set(xrange(prev_end, start))
            valid = True
            for index in between_set:
                if toks[index] not in unalign_symbols and (not isNum(toks[index])):
                    valid = False
                    new_spans.append((prev_start, prev_end))
                    prev_start, prev_end = start, end
                    break
            if valid:
                prev_end = end

        else:
            prev_start, prev_end = start, end
    if prev_end is not None:
        new_spans.append((prev_start, prev_end))
    return new_spans

def extractDates(file, template_file):
    tempt_map = loadTemplates(template_file)
    for line in open(file):
        if line.strip():
            #print line.strip()
            dates_in_line = []
            toks = line.strip().split()
            n_toks = len(toks)
            aligned = set()
            for start in xrange(n_toks):
                if start in aligned:
                    continue
                for length in xrange(n_toks+1, 0, -1):
                    end = start + length
                    if end > n_toks:
                        continue

                    span_set = set(xrange(start, end))
                    if len(span_set & aligned) != 0:
                        continue

                    curr_tempt = extractTemplate(' '.join(toks[start:end]))
                    if curr_tempt in tempt_map:
                        if curr_tempt == 'NUM1' or curr_tempt == 'NUM2' or curr_tempt == 'NUM3' or curr_tempt == 'DAY_ZONE':
                            continue
                        if curr_tempt == 'NUM6':
                            month = int(toks[start][2:4])
                            if month > 12:
                                continue
                            day = int(toks[start][4:])
                            if day > 31:
                                continue

                        dates_in_line.append((start, end))
                        aligned |= span_set
                        break
            #print dates_in_line
            dates_in_line = mergeDates(dates_in_line, toks)
            dates_in_line = ['%d-%d' % (start ,end) for (start, end) in dates_in_line]

            print ' '.join(dates_in_line)

if __name__ == '__main__':
    #extractAllDate(sys.argv[1])
    extractDates(sys.argv[1], './all_templates')
    #templates = defaultdict(int)
    #for line in open(sys.argv[1]):
    #    if line.strip():
    #        curr_tempt = extractTemplate(line.strip())
    #        templates[curr_tempt] += 1
    #all_templates = sorted(templates.items(), key=lambda x: -x[1])
    #for (tempt, num) in all_templates:
    #    print tempt, '####', num
