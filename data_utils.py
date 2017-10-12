# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import re
import codecs
import gensim
import random
import jieba
import numpy as np
from utils import iob2,iob_iobes
__author__='liuxiaoyu'

PAD,UNK,NUM='<PAD>','<UNK>','0'
SINGLE,BEGIN,INSIDE,END=0,1,2,3

def create_dict(item_list):
    assert type(item_list) is list
    dictionary=dict()
    for items in item_list:
        for item in items:
            dictionary[item]=dictionary.get(item,0)+1
    return dictionary

def create_mapping(dictionary):
    sorted_items=sorted(dictionary.items(),key=lambda x:(-x[-1],x[0]))
    id_to_item={i:item[0] for i,item in enumerate(sorted_items)}
    item_to_id={item:id for id,item in id_to_item.items()}
    return item_to_id,id_to_item

def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in BIES format
    """
    seg_feature=[]
    for word in jieba.cut(string):
        if len(word)==1:
            seg_feature.append(SINGLE)
        else:
            tmp=[INSIDE]*len(word)
            tmp[0]=BEGIN
            tmp[-1]=END
            seg_feature.extend(tmp)
    return seg_feature

def load_sentences(path,zeros=False):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences=[]
    sentence=[]
    for line in codecs.open(path,'r','utf-8'):
        line=line.rstrip()
        if zeros:
            line=re.sub('\d',NUM,line)
        if not line and len(sentence)>0:
            sentences.append(sentence)
            sentence=[]
        else:
            if line[0]==" ":
                line="$"+line[1:]
            word=line.split()
            assert len(word)>=2
            sentence.append(word)
    if len(sentence)>0:
        sentences.append(sentence)
    return sentences

def update_tag_scheme(sentences,tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    new_sentences=[]
    for i,sentence in enumerate(sentences):
        tags=[word[-1] for word in sentence]
        # check whether tagging scheme is IOB format or not
        new_tags = iob2(tags)
        if not new_tags:
            error_str='\n'.join([' '.join(word) for word in sentence])
            raise Exception("Sentence should be given in IOB format! "
                            "Please check sentence %i \n %s") % (i+1,error_str)
        # convert tagging scheme
        if tag_scheme=='iob':
            pass
        elif tag_scheme=='iobes':
            new_tags=iob_iobes(new_tags)
        else:
            raise Exception('Unknown tag scheme!')
        new_sentences.append([[word[0],tag] for word,tag in zip(sentence,new_tags)])
    return new_sentences

def char_mapping(sentences):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars=[[word[0] for word in sentence] for sentence in sentences]
    dictionary = create_dict(chars)
    dictionary[PAD]=1e9+1
    dictionary[UNK]=1e9
    char_to_id,id_to_char=create_mapping(dictionary)
    print "Found %d unique chars (%d in total)" % (len(char_to_id),sum(len(x) for x in chars))
    return char_to_id,id_to_char

def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]
    dictionary = create_dict(tags)
    tag_to_id, id_to_tag = create_mapping(dictionary)
    print("Found %i unique named entity tags" % len(dictionary))
    return tag_to_id, id_to_tag

def augment_with_pretrained(char_to_id, pre_emb_path):
    """
    Augment the dictionary with words that have a pretrained embedding.
    """
    pre_trained = gensim.models.KeyedVectors.load_word2vec_format(pre_emb_path)
    pre_trained_vocab = pre_trained.vocab
    for c in pre_trained_vocab:
        if c not in char_to_id:
            char_to_id[c]=len(char_to_id)
    id_to_char={v:k for k,v in char_to_id.items()}
    return char_to_id,id_to_char

def prepare_dataset(sentences,char_to_id,tag_to_id,train=True):
    """"
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    none_index=tag_to_id["O"]
    UNK_index=char_to_id[UNK]
    data=[]
    for sentence in sentences:
        string=[w[0] for w in sentence]
        chars=[char_to_id.get(w,UNK_index)for w in string]
        segs=get_seg_features("".join(string))
        if train:
            tags=[tag_to_id[w[-1]] for w in sentence]
        else:
            tags=[none_index for _ in sentence]
        data.append([string,chars,segs,tags])
    return data

def load_word2vec(pre_emb_path,char_to_id,char_dim=100):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    pre_trained_emb = gensim.models.KeyedVectors.load_word2vec_format(pre_emb_path)
    inin_emb=np.random.uniform(-0.5,0.5,[len(char_to_id),char_dim])
    for char in char_to_id:
        if char in pre_trained_emb:
            inin_emb[char_to_id[char]]=pre_trained_emb[char]
    inin_emb[char_to_id[PAD]]=np.zeros(shape=char_dim)
    return inin_emb

def strQ2B(ustring):
    """
    Convert full-width character to half-width one
    """
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring

def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)

class BatchManager(object):

    def __init__(self,data,batch_size,shuffle=True):
        self.data=data
        self.batch_size=batch_size
        self.num_batch=len(data)/batch_size
        self.shuffle = shuffle

    def _pad_data(self,data):
        max_length=max([len(sentence[0]) for sentence in data])
        pad_strings=[]
        pad_chars=[]
        pad_segs=[]
        pad_targets=[]
        for strings,chars,segs,targets in data:
            padding=[0]*(max_length-len(strings))
            pad_strings.append(strings+padding)
            pad_chars.append(chars+padding)
            pad_segs.append(segs+padding)
            pad_targets.append(targets+padding)
        return [pad_strings,pad_chars,pad_segs,pad_targets]

    def iter_batch(self):
        if self.shuffle:
            random.shuffle(self.data)
        for i in range(self.num_batch):
            batch_data=self.data[i*self.batch_size:(i+1)*self.batch_size]
            yield self._pad_data(batch_data)
        if not self.shuffle:
            batch_data=self.data[self.num_batch*self.batch_size:]
            yield self._pad_data(batch_data)


















