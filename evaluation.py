from __future__ import print_function
import os
import pickle

import numpy
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary  # NOQA
import torch
from model import VSE, order_sim
from collections import OrderedDict

# No change
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)

# no change
class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)

######################## chnage captions to english and german ###############################
def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    R_deen,R_ende,SIMIL_en,SIMIL_de = model.val_start()

    end = time.time()
    total = 0
    un = 0
    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        for caption in captions:
            total += len(caption)
            for i in range(len(caption)):
                if(caption[i] == 3 or caption[i] == 8197):
                    un += 1

        # compute the embeddings
        X,img_emb, cap_emb = model.forward_emb(images, captions, lengths, R_ende,R_deen,
                                             volatile=True)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        # measure accuracy and record loss
        model.forward_loss(R_deen, R_ende, X, img_emb, cap_emb)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions
    print(str(total)+' '+str(un))
    return R_deen,R_ende, SIMIL_en, SIMIL_de, img_embs, cap_embs


def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    if data_path is not None:
        opt.data_path = data_path
#############################################################################################
############################################################################################
    # load vocabulary used by the model
    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)
###############?????????????????
    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])
############################

#####################################################################################
######################################################################################
   # print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)
#######################################################################################

   # print('Computing results...')
    _,_,SIMIL_en,SIMIL_de,img_embs, cap_embs = encode_data(model, data_loader)
    #print('Images: %d, Captions: %d' %(img_embs.shape[0] / 10, cap_embs.shape[0]))
###############################################################################################
###############################################################################################
    if not fold5:
        # no cross-validation, full evaluation
        ############################## I2T and T2I for English ###################################
        r_en, r_de = i2t(img_embs, cap_embs, measure=opt.measure, return_ranks=False)
        ri_en, ri_de = t2i(img_embs, cap_embs, measure=opt.measure, return_ranks=False)
        
        ar_en = (r_en[0] + r_en[1] + r_en[2]) / 3
        ari_en = (ri_en[0] + ri_en[1] + ri_en[2]) / 3
        rsum_en = r_en[0] + r_en[1] + r_en[2] + ri_en[0] + ri_en[1] + ri_en[2]
        print("English I2T and T2I")
        print("rsum: %.1f" % rsum_en)
        print("Average i2t Recall: %.1f" % ar_en)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r_en)
        print("Average t2i Recall: %.1f" % ari_en)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri_en)
        
        ar_de = (r_de[0] + r_de[1] + r_de[2]) / 3
        ari_de = (ri_de[0] + ri_de[1] + ri_de[2]) / 3
        rsum_de = r_de[0] + r_de[1] + r_de[2] + ri_de[0] + ri_de[1] + ri_de[2]
        print("German I2T and T2I")
        print("rsum: %.1f" % rsum_de)
        print("Average i2t Recall: %.1f" % ar_de)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r_de)
        print("Average t2i Recall: %.1f" % ari_de)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri_de)
        print('similarity in (en-de) pairs: %.4f' %SIMIL_en)
        print('similarity in (de-en) pairs: %.4f' %SIMIL_de)
        ############################################################################################
    else:
    ###################### Since I use f30k, no change ##################################
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                         cap_embs[i * 5000:(i + 1) *
                                  5000], measure=opt.measure,
                         return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                           cap_embs[i * 5000:(i + 1) *
                                    5000], measure=opt.measure,
                           return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])
	print('similarity in (en-de) pairs: %.4f' %SIMIL_en)
	print('similarity in (de-en) pairs: %.4f' %SIMIL_de)
    ##########################################################################################

   # torch.save({'rt_en': rt_en,'rt_de':rt_de, 'rti_en':rti_en, 'rti_de': rti_de}, 'ranks.pth.tar')


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (10N, K) matrix of images
    Captions: (10N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / 10
    # index_list = []
    #print(npts)
    captions_en = numpy.zeros((int(len(captions)/2),captions.shape[1]),dtype = numpy.float64)
    captions_de = numpy.zeros((int(len(captions)/2),captions.shape[1]),dtype = numpy.float64)
    #print(captions.dtype)
    CCC = 0
    JJJ = 0
    while CCC < len(captions):
    	captions_en[JJJ:JJJ+5] = captions[CCC:CCC+5]
    	captions_de[JJJ:JJJ+5] = captions[CCC+5:CCC+10]
	CCC = CCC + 10
	JJJ = JJJ + 5

    ranks_en = numpy.zeros(npts)
    ranks_de = numpy.zeros(npts)


    for index in range(npts):

        # Get query image
        im = images[10 * index].reshape(1, images.shape[1])

        # Compute scores for english
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 10 * (index + bs))
                im2 = images[10 * index:mx:10]
                d2_en = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions_en).cuda())
                d2_en = d2_en.cpu().numpy()
            d_en = d2_en[index % bs]
        else:
            d_en = numpy.dot(im, captions_en.T).flatten()
        inds_en = numpy.argsort(d_en)[::-1]

        # Score for english
        rank_en = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp_en = numpy.where(inds_en == i)[0][0]
            if tmp_en < rank_en:
                rank_en = tmp_en
        ranks_en[index] = rank_en




        # Compute scores for german
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 10 * (index + bs))
                im2 = images[10 * index:mx:10]
                d2_de = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions_de).cuda())
                d2_de = d2_de.cpu().numpy()
            d_de = d2_de[index % bs]
        else:
            d_de = numpy.dot(im, captions_de.T).flatten()
        inds_de = numpy.argsort(d_de)[::-1]

        # Score for german
        rank_de = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp_de = numpy.where(inds_de == i)[0][0]
            if tmp_de < rank_de:
                rank_de = tmp_de
        ranks_de[index] = rank_de





    # Compute metrics
    #r@n means that the best ranking between captions has the rank below n
    r1_en = 100.0 * len(numpy.where(ranks_en < 1)[0]) / len(ranks_en)
    r5_en = 100.0 * len(numpy.where(ranks_en < 5)[0]) / len(ranks_en)
    r10_en = 100.0 * len(numpy.where(ranks_en < 10)[0]) / len(ranks_en)
    medr_en = numpy.floor(numpy.median(ranks_en)) + 1
    meanr_en = ranks_en.mean() + 1


    r1_de = 100.0 * len(numpy.where(ranks_de < 1)[0]) / len(ranks_de)
    r5_de = 100.0 * len(numpy.where(ranks_de < 5)[0]) / len(ranks_de)
    r10_de = 100.0 * len(numpy.where(ranks_de < 10)[0]) / len(ranks_de)
    medr_de = numpy.floor(numpy.median(ranks_de)) + 1
    meanr_de = ranks_de.mean() + 1


    return (r1_en, r5_en, r10_en, medr_en, meanr_en), (r1_de, r5_de, r10_de, medr_de, meanr_de)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (10N, K) matrix of images
    Captions: (10N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / 10

    #print('t2i npts'+str(npts))

    ims = numpy.array([images[i] for i in range(0, len(images), 10)])

    ranks_en = numpy.zeros(5 * npts)
    ranks_de = numpy.zeros(5 * npts)

    for index in range(npts):

        # Get query captions
        queries = captions[10 * index:10 * index + 10]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 10 * index % bs == 0:
                mx = min(captions.shape[0], 10 * index + bs)
                q2 = captions[10 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (10 * index) % bs:(10 * index) % bs + 10].T
        else:
            d = numpy.dot(queries, ims.T)
	#print('shape of queries'+str(d.shape))
        inds = numpy.zeros(d.shape)



	for i in range(len(inds)):
		if(i < 5):
         		inds[i] = numpy.argsort(d[i])[::-1]
         		ranks_en[5 * index + i] = numpy.where(inds[i] == index)[0][0]
		elif(i >= 5):
                	inds[i] = numpy.argsort(d[i])[::-1]
                	ranks_de[5 * index + i - 5] = numpy.where(inds[i] == index)[0][0]



    # Compute metrics
    #r@n means that the image you want is ranked below n
    r1_en = 100.0 * len(numpy.where(ranks_en < 1)[0]) / len(ranks_en)
    r5_en = 100.0 * len(numpy.where(ranks_en < 5)[0]) / len(ranks_en)
    r10_en = 100.0 * len(numpy.where(ranks_en < 10)[0]) / len(ranks_en)
    medr_en = numpy.floor(numpy.median(ranks_en)) + 1
    meanr_en = ranks_en.mean() + 1

    r1_de = 100.0 * len(numpy.where(ranks_de < 1)[0]) / len(ranks_de)
    r5_de = 100.0 * len(numpy.where(ranks_de < 5)[0]) / len(ranks_de)
    r10_de = 100.0 * len(numpy.where(ranks_de < 10)[0]) / len(ranks_de)
    medr_de = numpy.floor(numpy.median(ranks_de)) + 1
    meanr_de = ranks_de.mean() + 1

    return (r1_en, r5_en, r10_en, medr_en, meanr_en), (r1_de, r5_de, r10_de, medr_de, meanr_de)

