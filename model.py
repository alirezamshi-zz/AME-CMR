import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
import pickle

##################################################################################################

import io
import collections


def save_embedding(X,R_ende,R_deen,i):
    x_en = X[4:8194]
    x_de = X[8198:]
    #print(x_en.dtype)
    #print(R_ende.dtype)
    #torch.zerro
    result_en = np.dot(x_en,R_ende.T)
    result_de = np.dot(x_de,R_deen.T)
    np.save('Emb/emb'+str(i)+'en',result_en)
    np.save('Emb/emb'+str(i)+'de',result_de)
    i += 1
    return i

def compute_csls_accuracy(x_src, x_tgt, lexicon, lexicon_size=-1, k=10, bsz=1024):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())

    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8

    sr = x_src[list(idx_src)]
    sc = np.dot(sr, x_tgt.T)
    similarities = 2 * sc
    sc2 = np.zeros(x_tgt.shape[0])
    for i in range(0, x_tgt.shape[0], bsz):
        j = min(i + bsz, x_tgt.shape[0])
        sc_batch = np.dot(x_tgt[i:j, :], x_src.T)
        dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
        sc2[i:j] = np.mean(dotprod, axis=1)
    similarities -= sc2[np.newaxis, :]

    nn = np.argmax(similarities, axis=1).tolist()
    correct = 0.0
    for k in range(0, len(lexicon)):
        if nn[k] in lexicon[idx_src[k]]:
            correct += 1.0
    return correct / lexicon_size

def compute_nn_accuracy(x_src, x_tgt, lexicon, bsz=100, lexicon_size=-1):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())
    acc = 0.0
    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8
    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        scores = np.dot(x_tgt, x_src[idx_src[i:e]].T)
        pred = scores.argmax(axis=0)
        for j in range(i, e):
            if pred[j - i] in lexicon[idx_src[j]]:
                acc += 1.0
    #print(acc)
    return acc / lexicon_size

#def compute_nn_accuracy(x_src, x_tgt, lexicon, bsz=100, lexicon_size=-1):
#    if lexicon_size < 0:
#        lexicon_size = len(lexicon)
#    idx_src = list(lexicon.keys())
#    acc = 0.0
#    x_src = torch.from_numpy(x_src).cuda()
#    x_tgt = torch.from_numpy(x_tgt).cuda()
    
#    print('x_src')
#    print(x_src)
#    x_src /= torch.norm(x_src,2,1).unsqueeze(1) + 1e-8
#    x_tgt /= torch.norm(x_tgt,2,1).unsqueeze(1) + 1e-8
    #print(x_src)
#    for i in range(0, len(idx_src), bsz):
#        e = min(i + bsz, len(idx_src))
        #print(x_tgt)
        #print(x_src[idx_src[i:e]])
#        scores = torch.mm(x_tgt,x_src[idx_src[i:e]].t())
#        pred = scores.max(0)[1]
#        for j in range(i, e):
#            if pred[j - i] in lexicon[idx_src[j]]:
#                acc += 1.0
#		print(acc)
#    print(acc)
#    return acc*1.0 / lexicon_size
    

def load_lexicon(filename, words_src, words_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    lexicon = collections.defaultdict(set)
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        word_src, word_tgt = line.split()
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print("Coverage of source vocab: %.4f" % (coverage))
    return lexicon, float(len(vocab))

            
def idx(words):
    w2i = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
    return w2i
      
def LOAD_VECTORS(fname, maxload=200000, norm=True, center=False, verbose=True):
    if verbose:
        print("Loading vectors from %s" % fname)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    if maxload > 0:
        n = min(n, maxload)
    x = np.zeros([n, d])
    words = []
    for i, line in enumerate(fin):
        if i >= n:
            break
        tokens = line.rstrip().split(' ')
        words.append(tokens[0])
        v = np.array(tokens[1:], dtype=float)
        x[i, :] = v
    if norm:
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if center:
        x -= x.mean(axis=0)[np.newaxis, :]
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if verbose:
        print("%d word vectors loaded" % (len(words)))
    return words, x


###### MAIN ######
def METRIC_SIMIL(WORDS_known_en, WORDS_known_de,index_words_en, index_words_de, X, R_deen, R_ende,words_en_fast,x_en_fast,words_de_fast,x_de_fast,dicotest,kind):
    words_en = []
    words_de = []
    for i in range(len(WORDS_known_en)):
        words_en.append(WORDS_known_en[i])
    for i in range(len(WORDS_known_de)):
        words_de.append(WORDS_known_de[i])

    R_deen = R_deen.cpu().numpy()
    R_ende = R_ende.cpu().numpy()

    x_en = np.dot(X[index_words_en],R_ende.T)
    
    x_de = np.dot(X[index_words_de],R_deen.T)
    #x_en = X[index_words_en]
    #x_de = X[index_words_de]
    #print(x_en)
    #print(x_de)
    idx_en_fast , idx_de_fast = idx(words_en_fast), idx(words_de_fast)
   
    if(kind == 1):
        x_src = x_en
        words_src = words_en
        x_tgt = x_de
        words_tgt = words_de

        #x_src = x_en_fast
        #words_src = words_en_fast
        #x_tgt = x_de_fast
        #words_tgt = words_de_fast
        #for i in range(len(words_en)):
         #       if(words_en[i] in idx_en_fast):
         #               x_src[idx_en_fast[words_en[i]]] = x_en[i]
    
        #for i in range(len(words_de)):
        #        if(words_de[i] in idx_de_fast):
        #                x_tgt[idx_de_fast[words_de[i]]] = x_de[i]
    elif(kind == 2):
        x_src = x_de 
        words_src = words_de
        x_tgt = x_en
        words_tgt = words_en

        #x_src = x_de_fast
        #words_src = words_de_fast
        #x_tgt = x_en_fast
        #words_tgt = words_en_fast
   
        #for i in range(len(words_de)):
        #        if(words_de[i] in idx_de_fast):
        #                x_src[idx_de_fast[words_de[i]]] = x_de[i]

        #for i in range(len(words_en)):
        #        if(words_en[i] in idx_en_fast):
        #                x_tgt[idx_en_fast[words_en[i]]] = x_en[i]
    #else:
#       print("ERROR in kind selecting")
    

    #############################################################################
    src2tgt, lexicon_size = load_lexicon(dicotest, words_src, words_tgt)

    nn = compute_nn_accuracy(x_src, x_tgt, src2tgt, lexicon_size=lexicon_size)
    return nn

def FIND_SIMIL(WORDS_known_en, WORDS_known_de,matrix, R_deen, R_ende,words_en,x_en,words_de,x_de):

    index_words_en = np.load('index_words_en.npy')
    index_words_de = np.load('index_words_de.npy')
    dicotest = 'en-de-test.txt'
    SIMIL_en = METRIC_SIMIL(WORDS_known_en, WORDS_known_de, index_words_en, index_words_de,matrix, R_deen, R_ende,words_en,x_en,words_de,x_de,dicotest,1)
    dicotest = 'de-en-test.txt'
    SIMIL_de = METRIC_SIMIL(WORDS_known_en, WORDS_known_de,index_words_en, index_words_de,matrix,R_deen, R_ende, words_en,x_en,words_de,x_de,dicotest,2)
    return SIMIL_en,SIMIL_de


####################################################################################################
#No change
def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

#No change
def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='vgg19', use_abs=False, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if data_name.endswith('_precomp'):
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, use_abs, no_imgnorm)
    else:
        img_enc = EncoderImageFull(
            embed_size, finetune, cnn_type, use_abs, no_imgnorm)

    return img_enc

######################## Since I use precomp data, No change #################################
# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features
####################################################################################################

#No change
class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderText(nn.Module):

    def __init__(self, WORDS_en, WORDS_de, train_set_ende, train_set_deen, maxsup, vocab_size,dictionary, word_dim, embed_size, num_layers,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.dictionary = dictionary
	self.COUNTER = 0
        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        #print(vocab_size)
        #self.embed.weight.requires_grad = False
        ############################################################################
        words_en = []
        for i in range(4,len(WORDS_en)):
            words_en.append(WORDS_en[i])
        words_de = []
        for i in range(8198,8194+len(WORDS_de)):
            words_de.append(WORDS_de[i])

        x_en = dictionary[4:8194]
        x_de = dictionary[8198:]

        self.words_src_deen = words_de
        self.words_tgt_deen = words_en

        self.x_src_deen = x_de
        self.x_tgt_deen = x_en
        
        self.idx_src_deen = self.idx(self.words_src_deen)
        self.idx_tgt_deen = self.idx(self.words_tgt_deen)

        # load train bilingual lexicon
        self.pairs_deen = self.load_pairs(train_set_deen, self.idx_src_deen, self.idx_tgt_deen)
        if maxsup > 0 and maxsup < len(self.pairs_deen):
                self.pairs_deen = self.pairs_deen[:maxsup]

        # selecting training vector  pairs
        self.X_src_deen, self.Y_tgt_deen = self.select_vectors_from_pairs(self.x_src_deen, self.x_tgt_deen, self.pairs_deen)

        self.R_deen = torch.from_numpy(self.procrustes(self.X_src_deen, self.Y_tgt_deen)).float().cuda()

        self.R_deen.requires_grad = False
        
	#self.init_weigths_R_deen()
        self.words_src_ende = words_en
        self.words_tgt_ende = words_de

        self.x_src_ende = x_en
        self.x_tgt_ende = x_de

        self.idx_src_ende = self.idx(self.words_src_ende)
        self.idx_tgt_ende = self.idx(self.words_tgt_ende)

        # load train bilingual lexicon
        self.pairs_ende = self.load_pairs(train_set_ende, self.idx_src_ende, self.idx_tgt_ende)
        if maxsup > 0 and maxsup < len(self.pairs_ende):
                self.pairs_ende = self.pairs_ende[:maxsup]

        # selecting training vector  pairs
        self.X_src_ende, self.Y_tgt_ende = self.select_vectors_from_pairs(self.x_src_ende, self.x_tgt_ende, self.pairs_ende)
        
        self.R_ende = torch.from_numpy(self.procrustes(self.X_src_ende, self.Y_tgt_ende)).float().cuda()

        self.R_ende.requires_grad = False

	self.translator_ende = nn.Linear(word_dim,word_dim,bias = False)
	self.translator_ende.weight.requires_grad = False

        self.translator_deen = nn.Linear(word_dim,word_dim,bias = False)
        self.translator_deen.weight.requires_grad = False

	self.translator_I = nn.Linear(word_dim,word_dim,bias = False)
	self.translator_I.weight.requires_grad = False
	self.translator_I.weight.data.copy_(torch.eye(300))
        ############################################################################
        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)
       
        self.init_weights()


    def idx(self,words):
        w2i = {}
        for i, w in enumerate(words):
            if w not in w2i:
                w2i[w] = i
        return w2i

    def select_vectors_from_pairs(self,x_src, y_tgt, pairs):
        n = len(pairs)
        d = x_src.shape[1]
        x = np.zeros([n, d])
        y = np.zeros([n, d])
        for k, ij in enumerate(pairs):
            i, j = ij
            x[k, :] = x_src[i, :]
            y[k, :] = y_tgt[j, :]
        return x, y

    def load_pairs(self,filename, idx_src, idx_tgt, verbose=True):
        f = io.open(filename, 'r', encoding='utf-8')
        pairs = []
        tot = 0
        for line in f:
            a, b = line.rstrip().split(' ')
            tot += 1
            if a in idx_src and b in idx_tgt:
                pairs.append((idx_src[a], idx_tgt[b]))
        if verbose:
            coverage = (1.0 * len(pairs)) / tot
            print("Found pairs for training: %d - Total pairs in file: %d - Coverage of pairs: %.4f" % (len(pairs), tot, coverage))
        return pairs

    def procrustes(self,X_src, Y_tgt):
        U, s, V = np.linalg.svd(np.dot(Y_tgt.T, X_src))
        return np.dot(U, V)
        
    def init_weights(self):
        self.embed.weight.data.copy_(torch.from_numpy(self.dictionary))

    def forward(self,R_ende, R_deen, x, lengths):
        #print(x.data)
	mask = x.data >= 8194
        #print(mask)
        #print(mask.dtype)
	mask = 2*mask
	mask2 = x.data < 4
	mask_total = 2*(mask + mask2)
        #print(mask_total)
        mask3 = x.data >= 8194
        mask4 = x.data < 8198
        mask5 = mask3*mask4
        mask_total += mask5
        #print(mask_total)
        ############# 0< <4 --> 2, Eng --> 0, 8194< 8198< --> 5, De ---> 4####################
        mask_total = mask_total.unsqueeze(2).repeat(1,1,300).float()
	x = self.embed(x)

	self.translator_ende.weight.data.copy_(R_ende)
	self.translator_deen.weight.data.copy_(R_deen)
        #print('weight')
        #print(self.translator_ende.weight.is_cuda)
        #print('emb')
        #np.save('emb_after',self.translator_ende(self.embed.weight[4:8194]).cpu().numpy())
        #np.save('raw_emb_after',self.embed.weight[4:8194,:].cpu().numpy())
        #torch.ss
        #print(mask.requires_grad)
        #print(R_ende.requires_grad)
        #print(R_deen.requires_grad)
        #y = x.clone()
        x = mask_total*(5-mask_total)*(4-mask_total)/12*self.translator_I(x.clone())+(5-mask_total)*(2-mask_total)*(4-mask_total)/40*self.translator_ende(x.clone()) + mask_total*(mask_total-2)*(mask_total-4)/15*self.translator_I(x.clone())+mask_total*(5-mask_total)*(mask_total-2)/8*self.translator_deen(x.clone())

        #x1 = mask_total*(5-mask_total)*(4-mask_total)/12*self.translator_I(x)
        #x2 = (5-mask_total)*(2-mask_total)*(4-mask_total)/40*x.matmul(R_ende.t()) 
        #x3 = mask_total*(mask_total-2)*(mask_total-4)/15*self.translator_I(x)
        #x4 = mask_total*(5-mask_total)*(mask_total-2)/8*x.matmul(R_deen.t())
        #x = x1+x3
	#m,n,_ = x.shape
	#for i in range(m):
	#	for j in range(n):
	#		if(mask[i,j] == 0):
	#			x[i,j] = self.translator_ende(x[i,j].clone())
	#		else:
	#			x[i,j] = self.translator_deen(x[i,j].clone())
        packed = pack_padded_sequence(x, lengths, batch_first=True)
    
        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return self.embed.weight.data, out

#No change
#Im[a;b;c] and s[1;2;3]
#cosine = [a.1 a.2 a.3;b.1 b.2 b.3;c.1 c.2 c.3]
def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

#No change
#order = [||1-a|| ||2-a|| ||3-a||;||1-b|| ||2-b|| ||3-b||;||1-c|| ||2-c|| ||3-c||]
def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1)) - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

################# build a loss between captions like this##########################################




###############################################################################################
class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self,  WORDS_en,WORDS_de, train_set_deen, train_set_ende, maxsup, const, maxneg, knn, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        ##### I think this is the difference between ASYM and SYM similarity #################
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim
        ######################################################################################
        self.max_violation = max_violation

	self.const = const
	self.maxneg = maxneg
	self.knn = knn
	#####################################################################################
        words_en = []
        words_de = []
        for i in range(4,len(WORDS_en)):
            words_en.append(WORDS_en[i])
        for i in range(8198,8194+len(WORDS_de)):
            words_de.append(WORDS_de[i])

        self.words_src_deen = words_de
        self.words_tgt_deen = words_en
        

	self.idx_src_deen = self.idx(self.words_src_deen)
        self.idx_tgt_deen = self.idx(self.words_tgt_deen)

	# load train bilingual lexicon
	self.pairs_deen = self.load_pairs(train_set_deen, self.idx_src_deen, self.idx_tgt_deen)
	if maxsup > 0 and maxsup < len(self.pairs_deen):
    	    self.pairs_deen = self.pairs_deen[:maxsup]

	self.pairs_src_deen,self.pairs_tgt_deen = self.seperate_pairs(self.pairs_deen)


        self.words_src_ende = words_en
        self.words_tgt_ende = words_de
        
    
        self.idx_src_ende = self.idx(self.words_src_ende)
        self.idx_tgt_ende = self.idx(self.words_tgt_ende)
       
        # load train bilingual lexicon
        self.pairs_ende = self.load_pairs(train_set_ende, self.idx_src_ende, self.idx_tgt_ende)
        if maxsup > 0 and maxsup < len(self.pairs_ende):
            self.pairs_ende = self.pairs_ende[:maxsup]
	self.pairs_src_ende,self.pairs_tgt_ende = self.seperate_pairs(self.pairs_ende)

    def load_pairs(self,filename, idx_src, idx_tgt, verbose=True):
        f = io.open(filename, 'r', encoding='utf-8')
        pairs = []
        tot = 0
        for line in f:
            a, b = line.rstrip().split(' ')
            tot += 1
            if a in idx_src and b in idx_tgt:
                pairs.append((idx_src[a], idx_tgt[b]))
        if verbose:
            coverage = (1.0 * len(pairs)) / tot
            print("Found pairs for training: %d - Total pairs in file: %d - Coverage of pairs: %.4f" % (len(pairs), tot, coverage))
        return pairs
    def idx(self,words):
    	w2i = {}
    	for i, w in enumerate(words):
        	if w not in w2i:
            		w2i[w] = i
    	return w2i

    def getknn(self,sc, x, y, k=10):
	sidx = torch.topk(sc,k)[1]
        ytopk = y[sidx.cpu().numpy().flatten(), :]
        ytopk = ytopk.view(sidx.shape[0], sidx.shape[1], y.shape[1])
        f = float(torch.sum(sc[torch.arange(sc.shape[0]).int().unsqueeze(1).numpy(), sidx]))
        df = torch.mm(ytopk.sum(1).t(), x)
        return f/k ,df / k

    def rcsls(self,X_src, Y_tgt, Z_src, Z_tgt, R, knn=10):
        X_trans = torch.mm(X_src, R.t())
        f = 2 * float(torch.sum(X_trans * Y_tgt))
        df = 2 * torch.mm(Y_tgt.t(), X_src)
        fk0, dfk0 = self.getknn(torch.mm(X_trans, Z_tgt.t()), X_src, Z_tgt, knn)
        fk1, dfk1 = self.getknn(torch.mm(torch.mm(Z_src, R.t()), Y_tgt.t()).t(), Y_tgt, Z_src, knn)
        f = f - fk0 -fk1
        df = df - dfk0 - dfk1.t()
        return -f / X_src.shape[0], -df / X_src.shape[0]

    def select_vectors_from_pairs(self,x_src, y_tgt, pairs_src, pairs_tgt):
        n = len(pairs_src)
        d = x_src.shape[1]
        x = torch.zeros([n, d])
        y = torch.zeros([n, d])
	x = x_src[pairs_src]
        y = y_tgt[pairs_tgt]
        return x, y

    def seperate_pairs(self,pairs):
	pairs_src = np.zeros((1,len(pairs)))
	pairs_tgt = np.zeros((1,len(pairs)))
	for i in range(len(pairs)):
    		pairs_src[0,i] = pairs[i][0]
    		pairs_tgt[0,i] = pairs[i][1]
	return pairs_src,pairs_tgt

    def compute_rcsls(self,X,R,kind):

        x_en = X[4:8194]
        x_de = X[8198:]

        if(kind == 1):
            x_src = x_de
            x_tgt = x_en
        
            X_src, Y_tgt = self.select_vectors_from_pairs(x_src, x_tgt, self.pairs_src_deen,self.pairs_tgt_deen)
            X_src = X_src.cuda()
	    Y_tgt = Y_tgt.cuda()
	    # adding negatives for RCSLS
	    #Z_src = x_src[:self.maxneg, :]
	    #Z_tgt = x_tgt[:self.maxneg, :]
            Z_src = x_src
            Z_tgt = x_tgt

            f, df = self.rcsls(X_src,Y_tgt,Z_src, Z_tgt,R, self.knn)
            return f, df
	elif(kind == 2):

            x_src = x_en
            x_tgt = x_de

            X_src, Y_tgt = self.select_vectors_from_pairs(x_src, x_tgt, self.pairs_src_ende, self.pairs_tgt_ende)
            X_src = X_src.cuda()
	    Y_tgt = Y_tgt.cuda()
            # adding negatives for RCSLS
            #Z_src = x_src[:self.maxneg, :]
            #Z_tgt = x_tgt[:self.maxneg, :]
            Z_src = x_src
            Z_tgt = x_tgt

            f, df = self.rcsls(X_src,Y_tgt,Z_src, Z_tgt,R, self.knn)
            return f,df
	else:
	    print('Wrong kind for computing loss')
	    return
        ####################### get sentences from English and German##########################
    def forward(self,R_deen, R_ende, X, im, s):

	f_deen,df_deen = self.compute_rcsls(X,R_deen,1)
        f_ende,df_ende = self.compute_rcsls(X,R_ende,2)
        ######################### Compute image-sentence English ##############################
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        #######################################################################################
        #######################################################################################
        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
	#print(cost_s.sum() + cost_im.sum())
        #######################################################################################
        #######################################################################################
        # keep the maximum violating negative for each query
        #if self.max_violation:
        #    cost_s = cost_s.max(1)[0]
        #    cost_im = cost_im.max(0)[0]
        #######################################################################################
        return cost_s.sum() + cost_im.sum(),f_deen, f_ende, df_deen, df_ende

def save_vectors(fname, x, words):
    n, d = x.shape
    fout = io.open(fname, 'w', encoding='utf-8')
    fout.write(u"%d %d\n" % (n, d))
    for i in range(n):
        fout.write(words[i] + " " + " ".join(map(lambda a: "%.4f" % a, x[i, :])) + "\n")
    fout.close()

def save_vector(X,WORDS_en,WORDS_de,R_ende,R_deen):
    words_en = []
    words_de = []
    for i in range(4,len(WORDS_en)):
        words_en.append(WORDS_en[i])
    for i in range(4,len(WORDS_de)):
        words_de.append(WORDS_de[i+8194])
    
    x_en = np.dot(X[4:8194,:],R_ende.T)
    x_de = np.dot(X[8198:,:],R_deen.T)
    #words_en = []
    #words_de = []
    #for i in range(4,len(WORDS_en)):
    #    words_en.append(WORDS_en[i])
    #for i in range(8198,8194+len(WORDS_de)):
    #    words_de.append(WORDS_de[i])
    #x_en = np.dot(X[4:8194,:],R_ende.T)
    #x_de = np.dot(X[8198:,:],R_deen.T)
    save_vectors('myemb.en.vec',x_en,words_en)
    save_vectors('myemb.de.vec',x_de,words_de)


#def save_vector(X,WORDS_known_en,WORDS_known_de,R_ende,R_deen):
#    words_en = []
#    words_de = []
#    for i in range(len(WORDS_known_en)):
#        words_en.append(WORDS_known_en[i])
#    for i in range(len(WORDS_known_de)):
#        words_de.append(WORDS_known_de[i])

    #R_deen = R_deen.cpu().numpy()
    #R_ende = R_ende.cpu().numpy()

#    index_words_en = np.load('index_words_en.npy')
#    index_words_de = np.load('index_words_de.npy')

#    x_en = np.dot(X[index_words_en],R_ende.T)
#    x_de = np.dot(X[index_words_de],R_deen.T)
    #words_en = []
    #words_de = []
    #for i in range(4,len(WORDS_en)):
    #    words_en.append(WORDS_en[i])
    #for i in range(8198,8194+len(WORDS_de)):
    #    words_de.append(WORDS_de[i])
    #x_en = np.dot(X[4:8194,:],R_ende.T)
    #x_de = np.dot(X[8198:,:],R_deen.T)
#    save_vectors('myemb.en.vec',x_en,words_en)
#    save_vectors('myemb.de.vec',x_de,words_de)

def load_vectorsx(fname, maxload=200000, norm=True, center=False, verbose=True):
    if verbose:
        print("Loading vectors from %s" % fname)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    if maxload > 0:
        n = min(n, maxload)
    x = np.zeros([n, d])
    words = []
    for i, line in enumerate(fin):
        if i >= n:
            break
        tokens = line.rstrip().split(' ')
        words.append(tokens[0])
        v = np.array(tokens[1:], dtype=float)
        x[i, :] = v
    return words, x



class VSE(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models
	self.fold_deen, self.Rold_deen = 0, []
        self.fold_ende, self.Rold_ende = 0, []
        self.COUNTER = 0

        self.grad_clip = opt.grad_clip
	self.lr_translator = opt.lr_translator
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm)

	self.dic_loaded = np.load(opt.dictionary)


        self.vocab_size = opt.vocab_size

        with open(opt.WORDS_en, "rb") as fp:   # Unpickling
                self.WORDS_en = pickle.load(fp)

        with open(opt.WORDS_known_en, "rb") as fp:   # Unpickling
                self.WORDS_known_en = pickle.load(fp)

        with open(opt.WORDS_de, "rb") as fp:   # Unpickling
                self.WORDS_de = pickle.load(fp)

        with open(opt.WORDS_known_de, "rb") as fp:   # Unpickling
                self.WORDS_known_de = pickle.load(fp)


        self.txt_enc = EncoderText(self.WORDS_en, self.WORDS_de, opt.train_set_ende, opt.train_set_deen, opt.maxup,opt.vocab_size,self.dic_loaded, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs)
	##############################################################################################################

	self.words_en, self.x_en = LOAD_VECTORS('/mnt/storage01/alizera/mydata/datas/align/wiki.en.align.vec', maxload=200)
        self.words_de, self.x_de = LOAD_VECTORS('/mnt/storage01/alizera/mydata/datas/align/wiki.de.align.vec', maxload=200)
	#self.words_en, self.x_en = LOAD_VECTORS('/mnt/storage01/alizera/mydata/datas/align/wiki.en.align.vec', maxload=200000)
	#self.words_de, self.x_de = LOAD_VECTORS('/mnt/storage01/alizera/mydata/datas/align/wiki.de.align.vec', maxload=200000)
        ####################################################################################
        if torch.cuda.is_available():
            self.img_enc.cuda()
        ######################################################################
            self.txt_enc.cuda()
        ######################################################################
            cudnn.benchmark = True

        # Loss and Optimizer
        #self.ende_train = 'en-de-train.txt.'
        #self.ende_test = 'de-en-test.txt'
        self.criterion = ContrastiveLoss(self.WORDS_en,self.WORDS_de, opt.train_set_deen,opt.train_set_ende, opt.maxup, opt.const, opt.maxneg, opt.knn, margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)
        ########################################################################
        params = list(filter(lambda p: p.requires_grad, self.txt_enc.parameters()))
	#print(params)
	self.R_deen = self.txt_enc.R_deen
        self.R_ende = self.txt_enc.R_ende
        ########################################################################
        params += list(self.img_enc.fc.parameters())
        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        ############################################
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        ############################################
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        #######################################################
        #save_vector(state_dict[1]['embed.weight'].cpu().numpy(),self.WORDS_en,self.WORDS_de)
        #torch.sss
        #self.R_ende = torch.from_numpy(np.load('R_ende.npy')).float().cuda()
        #print(state_dict[1]['embed.weight'].is_cuda)
        temp = np.zeros((self.vocab_size,300),dtype = np.float32)
        temp[0:4,:] = state_dict[1]['embed.weight'].cpu().numpy()[0:4,:]
        temp[4:8194,:] = np.dot(state_dict[1]['embed.weight'].cpu().numpy()[4:8194,:],np.load('R_ende.npy'))
        #print(temp[4:8194,:][0])
        np.save('emb_before',temp[4:8194,:])
        np.save('raw_emb_before',state_dict[1]['embed.weight'][4:8194,:].cpu().numpy())
        temp[8194:,:] = state_dict[1]['embed.weight'].cpu().numpy()[8194:,:]
        #print(state_dict[1]['embed.weight'][4:8194])
        #temp = state_dict[1]['embed.weight'].cpu().numpy()
        #state_dict[1]['embed.weight'] = torch.from_numpy(temp).cuda()
        self.txt_enc.load_state_dict(state_dict[1])
	self.R_deen = torch.from_numpy(np.load('R_deen.npy')).cuda()
	self.R_ende = torch.from_numpy(np.load('R_ende.npy')).cuda()
        #print(self.R_ende)
        #self.R_ende = torch.from_numpy(np.eye(300)).float().cuda()
	#save_vector(state_dict[1]['embed.weight'].cpu().numpy(),self.WORDS_en,self.WORDS_de,self.R_ende.cpu().numpy(),self.R_deen.cpu().numpy())
	#np.save('after_train.npy',list(self.txt_enc.parameters())[0].cpu().data.numpy())
        #######################################################
    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        #########################################
        self.txt_enc.train()
        #########################################

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        #########################################
        self.txt_enc.eval()
	matrix = list(self.txt_enc.parameters())[0].cpu().data.numpy()
	#print(matrix)
        #self.COUNTER = np.save('Emb/emb'+str(self.COUNTER),matrix)
        #self.COUNTER += 1
        self.COUNTER = save_embedding(matrix,self.R_ende.cpu().numpy(),self.R_deen.cpu().numpy(),self.COUNTER)
	simil_en,simil_de = FIND_SIMIL(self.WORDS_known_en, self.WORDS_known_de, matrix, self.R_deen, self.R_ende, self.words_en, self.x_en, self.words_de, self.x_de)
        #print(self.R_ende)
	return self.R_deen, self.R_ende, simil_en, simil_de
        #########################################

################################ Embeddings Caption for English and German ##############################
############### Change the input and return part#########################################################
    def forward_emb(self, images, captions, lengths,R_ende, R_deen, volatile=False):
        """Compute the image and caption embeddings
        """
        #####################################################################################################
        # Set mini-batch dataset
        if(volatile == True):
		with torch.no_grad():
        		images = Variable(images)
        		captions = Variable(captions)
        		if torch.cuda.is_available():
            			images = images.cuda()
            			###################################
            			captions = captions.cuda()
            			###################################
        		# Forward
        		img_emb = self.img_enc(images)
        		#####################################
        		X, cap_emb = self.txt_enc(R_ende, R_deen, captions, lengths)
        		return X, img_emb, cap_emb
	else:
                images = Variable(images)
                captions = Variable(captions)
                if torch.cuda.is_available():
                    images = images.cuda()
                    ###################################
                    captions = captions.cuda()
                    ###################################
                # Forward
                img_emb = self.img_enc(images)
                #####################################
                X, cap_emb = self.txt_enc(R_ende, R_deen, captions, lengths)
                return X, img_emb, cap_emb


    #################### change the input###################################
    def forward_loss(self, R_deen, R_ende, X, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        ##### criterian?????????#############
        loss, f_deen, f_ende, df_deen, df_ende = self.criterion(R_deen, R_ende, X, img_emb, cap_emb)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        return loss, f_deen, f_ende, df_deen, df_ende
    ############## change input
    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        
        X, img_emb, cap_emb = self.forward_emb(images, captions, lengths, self.R_ende,self.R_deen)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss,f_deen, f_ende,  df_deen,df_ende = self.forward_loss(self.R_deen,self.R_ende, X , img_emb, cap_emb)
         

        self.R_deen -= self.lr_translator * df_deen
        if f_deen > self.fold_deen:
            self.lr_translator /= 2
            f_deen, self.R_deen = self.fold_deen, self.Rold_deen

	self.fold_deen, self.Rold_deen = f_deen, self.R_deen


	self.R_ende -= self.lr_translator * df_ende
        if f_ende > self.fold_ende:
            self.lr_translator /= 2
            f_ende, self.R_ende = self.fold_ende, self.Rold_ende
        
        self.fold_ende, self.Rold_ende = f_ende, self.R_ende
        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

