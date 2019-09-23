from __future__ import print_function
from gensim.models import KeyedVectors
#from gensim.models.wrappers import FastText
import numpy as np
#import nltk
import pickle
from collections import Counter
from pycocotools.coco import COCO
import json
import argparse
import os
"""annotations = {
    'coco_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'coco': ['annotations/captions_train2014.json',
             'annotations/captions_val2014.json'],
    'f8k_precomp': ['train_caps.txt', 'dev_caps.txt'],
    '10crop_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'f30k_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'f8k': ['dataset_flickr8k.json'],
    'f30k': ['dataset_flickr30k.json'],
}"""
############################ Change train and dev to English and German versions##################
annotations_en = {
    'f30k_precomp' : ['train_en_caps.txt']
}
annotations_de = {
    'f30k_precomp' : ['train_de_caps.txt']
}


#################################################################################################
class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx_en = {}
        self.idx2word_en = {}

        self.word2idx_de = {}
        self.idx2word_de = {}

        self.idx = 0

	self.idx2word_known_en = {}
	self.idx_known_en = 0

        self.idx2word_known_de = {}
        self.idx_known_de = 0

	self.ID = 0
	self.ID_unknown = 0
        self.dictionary = np.zeros((200000,300),dtype = np.float32)
        self.index_words_known_en = np.zeros(50000,dtype = np.int32)
        self.index_words_known_de = np.zeros(50000,dtype = np.int32)

    def add_word(self, word,model,kind):
        if(kind == 1):
            if word not in self.word2idx_en:
                self.ID += 1
                self.word2idx_en[word] = self.idx
                self.idx2word_en[self.idx] = word
                try:
                    self.dictionary[self.idx,:] = model[word]
                    self.index_words_known_en[self.idx_known_en] = self.idx
                    self.idx2word_known_en[self.idx_known_en] = word
                    self.idx_known_en += 1
                except:
                    self.dictionary[self.idx,:] = np.random.normal(0,0.05,(1,300))
                    self.ID_unknown += 1
                self.idx += 1
        elif(kind == 2):
            if word not in self.word2idx_de:
	        self.ID += 1
                self.word2idx_de[word] = self.idx
                self.idx2word_de[self.idx] = word
                try:
		    self.dictionary[self.idx,:] = model[word]
                    self.index_words_known_de[self.idx_known_de] = self.idx
		    self.idx2word_known_de[self.idx_known_de] = word
		    self.idx_known_de += 1
                except:
		    self.dictionary[self.idx,:] = np.random.normal(0,0.05,(1,300))
		    self.ID_unknown += 1
	        self.idx += 1


    def __call__(self, word, kind):
        if(kind == 1):
            if word not in self.word2idx_en:
                return self.word2idx_en['<unk>']
            return self.word2idx_en[word]
        elif(kind == 2):
            if word not in self.word2idx_de:
                return self.word2idx_de['<unk>']
            return self.word2idx_de[word]

    def ratio(self):
        return self.ID_unknown*1.0/self.ID

    def WORDS_known_en(self):
        return self.idx2word_known_en

    def WORDS_known_de(self):
        return self.idx2word_known_de

    def WORDS_en(self):
	return self.idx2word_en

    def WORDS_de(self):
        return self.idx2word_de

    def __len__(self):
        return self.idx


def from_coco_json(path):
    coco = COCO(path)
    ids = coco.anns.keys()
    captions = []
    for i, idx in enumerate(ids):
        captions.append(str(coco.anns[idx]['caption']))

    return captions

def from_flickr_json(path):
    dataset = json.load(open(path, 'r'))['images']
    captions = []
    for i, d in enumerate(dataset):
        captions += [str(x['raw']) for x in d['sentences']]

    return captions

import io
def load_vectors(fname):
  fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
  n, d = map(int, fin.readline().split())
  data = {}
  idx=0
  for line in fin:
    #idx+=1    
    #if idx>5000:
    	#break        
    tokens = line.rstrip().split(' ')
    data[tokens[0]] = map(float, tokens[1:])
  return data


def from_txt(txt):
    captions = []
    #f = io.open(txt, 'r', encoding='utf-8', newline='\n', errors='ignore')
    with open(txt, 'rb') as f:
    	for line in f:
    		captions.append(line.strip())
    return captions

vocab = Vocabulary()
def build_vocab(kind, model_path ,data_path, data_name, jsons, threshold):
    """Build a simple vocabulary wrapper."""
    global vocab
    counter = Counter()
    for path in jsons[data_name]:
        full_path = os.path.join(os.path.join(data_path, data_name), path)
        if data_name == 'coco':
            captions = from_coco_json(full_path)
        elif data_name == 'f8k' or data_name == 'f30k':
            captions = from_flickr_json(full_path)
        else:
            captions = from_txt(full_path)
	#print(captions)
        for i, caption in enumerate(captions):
            #tokens = nltk.tokenize.word_tokenize(caption.lower().decode('utf-8'))
	    #print(caption)
            tokens = caption.decode('utf-8').split()
	    #tokens = caption.rstrip().split(' ')
	    #print(tokens)
	    counter.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    #print(words)
    # Create a vocab wrapper and add some special tokens.
    model = load_vectors(model_path)
    #model = KeyedVectors.load_word2vec_format(model_path,limit = 300000)
    vocab.add_word('<pad>',model,kind)
    vocab.add_word('<start>',model,kind)
    vocab.add_word('<end>',model,kind)
    vocab.add_word('<unk>',model,kind)
    for i, word in enumerate(words):
    	vocab.add_word(word,model,kind)
    del model


def main(data_path, data_name, model_path_en, model_path_de):
    ###############################################################################
    build_vocab(1,model_path_en,data_path, data_name, jsons=annotations_en, threshold=4)
    LEN = len(vocab)
    build_vocab(2,model_path_de,data_path, data_name, jsons=annotations_de, threshold=4)
    with open('/mnt/storage01/alizera/mydata/align/task2_resnet3/vocab/%s_vocab.pkl' % data_name, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("Saved vocabulary file to ", './vocab/%s_vocab.pkl' % data_name)
    np.save('/mnt/storage01/alizera/mydata/align/task2_resnet3/vocab/dictionary.npy',vocab.dictionary[0:len(vocab)])
    print("Saved dictionary file to ", './vocab/%s_dictionary.npy' % data_name)

    with open('/mnt/storage01/alizera/mydata/align/task2_resnet3/vocab/WORDS_en.txt', "wb") as fp:   #Pickling
	pickle.dump(vocab.WORDS_en(), fp)

    with open('/mnt/storage01/alizera/mydata/align/task2_resnet3/vocab/WORDS_de.txt', "wb") as fp:   #Pickling
        pickle.dump(vocab.WORDS_de(), fp)

    with open('/mnt/storage01/alizera/mydata/align/task2_resnet3/vocab/WORDS_knwon_en.txt', "wb") as fp:   #Pickling
        pickle.dump(vocab.WORDS_known_en(), fp)

    with open('/mnt/storage01/alizera/mydata/align/task2_resnet3/vocab/WORDS_knwon_de.txt', "wb") as fp:   #Pickling
        pickle.dump(vocab.WORDS_known_de(), fp)

    np.save('index_words_en.npy',vocab.index_words_known_en[0:vocab.idx_known_en])
    np.save('index_words_de.npy',vocab.index_words_known_de[0:vocab.idx_known_de])
    
    print('ratio of unknown words')
    print(vocab.ratio())
    print("len is:")
    print(LEN)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/mnt/storage01/alizera/mydata/align/task2_resnet3/data')
    
    parser.add_argument('--model_path_en', default='/mnt/storage01/alizera/mydata/datas/align/wiki.en.align.vec')
    parser.add_argument('--model_path_de', default='/mnt/storage01/alizera/mydata/datas/align/wiki.de.align.vec')
    #parser.add_argument('--model_path_en', default='myalign.en.sym.vec')
    #parser.add_argument('--model_path_de', default='myalign.de_with_tr.sym.vec')

    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    opt = parser.parse_args()
    main(opt.data_path, opt.data_name, opt.model_path_en, opt.model_path_de)

