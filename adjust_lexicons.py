import io
import argparse
import pickle

def idx(words):

	w2i = {}
	for i, w in enumerate(words):
		if w not in w2i:
			w2i[w] = i
	return w2i


def adjust_lexicon(directory, words_src, words_tgt, fname_save, verbose=True):
	'''
	This function gets the (combined) lexicons of MUSE benchmark, and adjusts them to the dataset.

	Inputs:
		directory: Path to train/test lexicons 
		words_src: source words for the first argument of lexicon pairs
		words_tgt: target words for the second argument of lexicon pairs
		fname_save: name of the file to be saved

	'''    
	# read the lexicons
	f = io.open(directory, 'r', encoding='utf-8')
    
	# open the file to save the adjusted lexicons
	fout = io.open(fname_save, 'w', encoding='utf-8')

	# save the lexicons which appear in both source and target words
	idx_src , idx_tgt = idx(words_src), idx(words_tgt)
	vocab_in = 0
	vocab_all = 0
	for line in f:
		word_src, word_tgt = line.split()
		vocab_all += 1
		if word_src in idx_src and word_tgt in idx_tgt:
			fout.write(word_src+" "+word_tgt+"\n")
			vocab_in += 1
	# show the coverage of lexicons
	fout.close()
	if verbose:
		coverage = vocab_in*1.0 / vocab_all
		print("Coverage of source vocab for %s: %.4f" % (fname_save,coverage))

def combine(directory, mode):
	'''
	This function combines the full and test lexicons of MUSE benchmark, and saves it to be used for building the train lexicons
	
	Inputs:
		directory: Path to lexicon pairs
		mode: define the kind of lexicon
	'''

	# read full lexicon file of MUSE benchmark
	lexicons1 = []
	with open(directory+mode+'.txt', 'r') as f:
		for line in f:
			lexicons1.append(line.strip())
	print(len(lexicons1))

	# read test lexicon file of MUSE benchmark
	lexicons2 = []
	with open(directory+mode+'.5000-6500.txt', 'r') as f:
		for line in f:
			lexicons2.append(line.strip())
	print(len(lexicons2))

	# combine and save the desired lexicon file
	save = open(directory+mode+'-combined.txt','w')

	for i in range(len(lexicons1)):
		save.write(lexicons1[i]+"\n")

	for i in range(len(lexicons2)):
		save.write(lexicons2[i]+"\n")
	save.close()



def main():
	# Input Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--en_de_dir', default='en-de/',help='Path to (en-de) lexicons')
	parser.add_argument('--de_en_dir', default='de-en/',help='Path to (de-en) lexicons')
	parser.add_argument('--en_ja_dir', default='en-ja/',help='Path to (en-ja) lexicons')
	parser.add_argument('--ja_en_dir', default='ja-en/',help='Path to (ja-en) lexicons')
	parser.add_argument('--en_multi30k', default='en_multi30k.txt',help='Path to English words of Multi30k dataset')
	parser.add_argument('--en_mscoco', default='en_mscoco.txt',help='Path to English words of MS-COCO dataset')
	parser.add_argument('--de_multi30k', default='de_multi30k.txt',help='Path to German words of Multi30k dataset')
	parser.add_argument('--ja_mscoco', default='ja_mscoco.txt',help='Path to Japanese words of MS-COCO dataset')
	parser.add_argument('--offset_multi30k', default='8194',help='Divide English and German words (if you have one file for words)')
	parser.add_argument('--offset_mscoco', default='11389',help='Divide English and Japanese words (if you have one file for words)')
    
	opt = parser.parse_args()
	print(opt)

	# read English and German words of Multi30k data
	with open(opt.en_multi30k, "rb") as fp:   # Unpickling
    		WORDS_en1 = pickle.load(fp)
    
	with open(opt.de_multi30k, "rb") as fp:   # Unpickling
    		WORDS_de = pickle.load(fp)
	words_en_multi30k = []
	words_de_multi30k = []

	for i in range(len(WORDS_en1)):
		words_en_multi30k.append(WORDS_en1[i])
  
	for i in range(4,len(WORDS_de)):
		words_de_multi30k.append(WORDS_de[opt.offset_multi30k+i])
	
	# read English and Japanese words of MS-COCO data
	with open(opt.en_mscoco, "rb") as fp:   # Unpickling
    		WORDS_en2 = pickle.load(fp)
    
	with open(opt.ja_mscoco, "rb") as fp:   # Unpickling
    		WORDS_ja = pickle.load(fp)
	words_en_mscoco = []
	words_ja_mscoco = []

	for i in range(len(WORDS_en2)):
		words_en_mscoco.append(WORDS_en2[i])
  
	for i in range(4,len(WORDS_ja)):
		words_ja_mscoco.append(WORDS_ja[opt.offset_mscoco+i])


	# building the train/test lexicons of Multi30k  
	combine(opt.en_de_dir, 'en-de')
	combine(opt.de_en_dir, 'de-en')

	adjust_lexicon(opt.en_de_dir+"en-de-combined.txt",words_en_multi30k,words_de_multi30k,"en-de-train")
	adjust_lexicon(opt.en_de_dir+"en-de.0-5000.txt",words_en_multi30k,words_de_multi30k,"en-de-test")

	adjust_lexicon(opt.de_en_dir+"de-en-combined.txt",words_de_multi30k,words_en_multi30k,"de-en-train")
	adjust_lexicon(opt.de_en_dir+"de-en.0-5000.txt",words_de_multi30k,words_en_multi30k,"de-en-test")	


	# building the train/test/ lexicons of MS-COCO
	combine(opt.en_ja_dir, 'en-ja')
	combine(opt.ja_en_dir, 'ja-en')

	adjust_lexicon(opt.en_ja_dir+"en-ja-combined.txt",words_en_mscoco,words_ja_mscoco,"en-ja-train")
	adjust_lexicon(opt.en_ja_dir+"en-ja.0-5000.txt",words_en_mscoco,words_ja_mscoco,"en-ja-test")

	adjust_lexicon(opt.ja_en_dir+"ja-en-combined.txt",words_ja_mscoco,words_en_mscoco,"ja-en-train")
	adjust_lexicon(opt.ja_en_dir+"ja-en.0-5000.txt",words_ja_mscoco,words_en_mscoco,"ja-en-test")	

main()


