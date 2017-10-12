# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import codecs
from conlleval import return_report
import data_utils

PAD,UNK,NUM='<PAD>','<UNK>','0'
SINGLE,BEGIN,INSIDE,END=0,1,2,3

def iob2(tags):
    """
    Convert IOB1 to IOB2, and default IOB format is IOB2
    """
    for i,tag in enumerate(tags):
        if tag=='O':
            continue
        split=tag.split('-')
        if len(split)!=2 or split[0] not in ['B','I']:
            return False
        if split[0]=='B':
            continue
        elif i==0 or tags[i-1]=='O':
            tags[i]='B'+tag[1:]
        elif tags[i-1][1:]==tag[1:]:
            continue
        else:
            tags[i]='B'+tag[1:]
    return tags

def iob_iobes(tags):
    """
    Convert IOB to IOBES
    """
    new_tags=[]
    for i,tag in enumerate(tags):
        if tag=='O':
            new_tags.append(tag)
        elif tag.split('-')[0]=='B':
            if (i+1)!=len(tags) and tags[i+1].split('-')[0]=='I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-','S-'))
        elif tag.split('-')[0]=='I':
            if (i+1)!=len(tags) and tags[i+1].split('-')[0]=='I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-','E-'))
        else:
            raise Exception('Invalid format!')
    return new_tags

def iobes_iob(tags):
    """
    Convert IOBES to IOB
    """
    new_tags=[]
    for i,tag in enumerate(tags):
        if tag.split('-')[0]=='B' or tag.split('-')[0]=='O':
            new_tags.append(tag)
        elif tag.split('-')[0]=='I':
            new_tags.append(tag)
        elif tag.split('-')[0]=='S':
            new_tags.append(tag.replace('S-','B-'))
        elif tag.split('-')[0]=='E':
            new_tags.append(tag.replace('E-','I-'))
        else:
            raise Exception('Invalid format!')
    return new_tags


def report_results(strings,predicts,goldens,id_to_char,id_to_tag,output_path,verbose=False):
    results=[]
    for i in range(len(strings)):
        result = []
        string = [x for x in strings[i]]
        pred = iobes_iob([id_to_tag[int(x)] for x in predicts[i]])
        gold = iobes_iob([id_to_tag[int(x)] for x in goldens[i]])
        for char, gold, pred in zip(string, gold, pred):
            result.append(" ".join([char, gold, pred]))
        results.append(result)

    with codecs.open(output_path,'w','utf-8') as f:
        for sentence in results:
            for line in sentence:
                f.write(line+'\n')
            f.write('\n')

    eval_lines =return_report(output_path)

    if verbose:
        for line in eval_lines[1:]:
            print line.strip()

    f1=float(eval_lines[1].strip().split()[-1])
    return f1

def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = data_utils.strQ2B(line)
    line = data_utils.replace_html(line)
    inputs = []
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id[UNK]
                   for char in line]])
    inputs.append([data_utils.get_seg_features(line)])
    inputs.append([[]])
    return inputs

def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item

def print_result_json(result_json):
    results=[]
    for entity in result_json['entities']:
        results.append(entity['word']+'/'+entity['type'])
    print 'Entities:'+' '.join(results)

if __name__ == '__main__':
    import jieba
    s=jieba.cut('复旦大学')









