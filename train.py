import pickle
import os
import time
import shutil

import torch

import data
from vocab import Vocabulary  # NOQA
from model import VSE
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data
import numpy as np
import logging
import tensorboard_logger as tb_logger

import argparse


def main():

    global SIMIL_EN_LIST
    global SIMIL_DE_LIST
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/mnt/storage01/alizera/mydata/align/task2_resnet3/data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--vocab_path', default='/mnt/storage01/alizera/mydata/align/task2_resnet3/vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=0.0001565, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='/mnt/storage01/alizera/mydata/align/task2_resnet3/runs/run_asym_whole_stg2_translator_lr0.0001565_lr12_testlast_edited_knn10_noabs',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='resnet152',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true',
                        help='Ensure the training is always done in '
                        'train mode (Not recommended).')
    parser.add_argument('--dictionary', default='/mnt/storage01/alizera/mydata/align/task2_resnet3/vocab/dictionary.npy',
                        help='Path to dictionary')
    parser.add_argument('--WORDS_en', default='/mnt/storage01/alizera/mydata/align/task2_resnet3/vocab/WORDS_en.txt',
                        help='Path to words of dictionary')
    parser.add_argument('--WORDS_de', default='/mnt/storage01/alizera/mydata/align/task2_resnet3/vocab/WORDS_de.txt',
                        help='Path to words of dictionary')
    parser.add_argument('--WORDS_known_en', default='/mnt/storage01/alizera/mydata/align/task2_resnet3/vocab/WORDS_knwon_en.txt',
                        help='Path to words of dictionary')
    parser.add_argument('--WORDS_known_de', default='/mnt/storage01/alizera/mydata/align/task2_resnet3/vocab/WORDS_knwon_de.txt',
                        help='Path to words of dictionary')
    parser.add_argument('--train_set_ende', default='en-de-train.txt',
                        help='path to train set of (en-de) pairs')
    parser.add_argument('--test_set_ende', default='en-de-test.txt',
                        help='Path to test set of (en-de) pairs')
    parser.add_argument('--train_set_deen', default='de-en-train.txt',
                        help='Path to train set of (de-en) pairs')
    parser.add_argument('--test_set_deen', default='de-en-test.txt',
                        help='Path to test set of (de-en) pairs')
    parser.add_argument('--maxup', default=-1,type = int,
                        help='Maximum number of training examples')
    parser.add_argument('--maxneg', default=20000,type = int,
                        help='Maximum number of negatives for the Extended RCSLS')
    parser.add_argument('--knn', default=10,type = int,
                        help='Number of nearest neighbour for RCSLS')
    parser.add_argument('--const', default=100,type = int,
                        help='Const to multiply in loss calculation')
    parser.add_argument('--lr_translator', default=12,type = int,
                        help='Learning rate for translator')
    opt = parser.parse_args()
    print(opt)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)
    #############################################################################################

    # Load data loaders
    ###################################################################################################
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.crop_size, opt.batch_size, opt.workers, opt)
    ###################################################################################################
    # Construct the model
    model = VSE(opt)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum_en = checkpoint['best_rsum_en']
	    best_rsum_de = checkpoint['best_rsum_de']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum_en {}, best_rsum_de {})"
                  .format(opt.resume, start_epoch, best_rsum_en, best_rsum_de))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    #####################
    best_rsum = 0
    #best_rsum_de = 0
    sim_in_model_en = []
    sim_in_model_de = []
    #####################
    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)
	
	if(epoch == 0):
		validate(opt, val_loader, model)
        # train for one epoch
        best_rsum = train(opt, train_loader, model, epoch, val_loader,best_rsum)

        # evaluate on validation set
        rsum_en, rsum_de, sim_en,sim_de,R_deen,R_ende = validate(opt, val_loader, model)
        ##################################################################
        rsum = rsum_en + rsum_de
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        
        #is_best_de = rsum_de > best_rsum_de
        #best_rsum_de = max(rsum_de, best_rsum_de)
	#if is_best_en or is_best_de:
            #sim_in_model_en.append(sim_en)
            #sim_in_model_de.append(sim_de)
            #logging.info("similarity in (en-de) pair: %.4f"%sim_en)
            #logging.info("similarity in (de-en) pair: %.4f"%sim_de)
        ##################################################################
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            #############################################
            'best_rsum': best_rsum,
            #'best_rsum_de': best_rsum_de,
            #############################################
            'opt': opt,
            'Eiters': model.Eiters,
        }, R_deen,R_ende, is_best, prefix=opt.logger_name + '/')

    with open('simil_en.txt', 'w') as f:
        for item in SIMIL_EN_LIST:
                f.write("%s\n" % item)
    with open('simil_de.txt', 'w') as f:
        for item in SIMIL_DE_LIST:
                f.write("%s\n" % item)  

    with open('simil_en_bestmodel.txt', 'w') as f:
        for item in sim_in_model_en:
                f.write("%s\n" % item)
    with open('simil_de_bestmodel.txt', 'w') as f:
        for item in sim_in_model_de:
                f.write("%s\n" % item)


def train(opt, train_loader, model, epoch, val_loader,best_rsum):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        if opt.reset_train:
            # Always reset to train mode, this is not the default behavior
            model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            rsum_en, rsum_de, sim_en,sim_de,R_deen,R_ende = validate(opt, val_loader, model)
            ##################################################################
            rsum = rsum_en + rsum_de
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)

            #is_best_de = rsum_de > best_rsum_de
            #best_rsum_de = max(rsum_de, best_rsum_de)
            #if is_best_en or is_best_de:
                #sim_in_model_en.append(sim_en)
                #sim_in_model_de.append(sim_de)
                #logging.info("similarity in (en-de) pair: %.4f"%sim_en)
                #logging.info("similarity in (de-en) pair: %.4f"%sim_de)       
            ##################################################################
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                #############################################
                'best_rsum': best_rsum,
                #'best_rsum_de': best_rsum_de,
                #############################################
                'opt': opt,
                'Eiters': model.Eiters,
            }, R_deen,R_ende, is_best, prefix=opt.logger_name + '/')

    return best_rsum



SIMIL_EN_LIST = []
SIMIL_DE_LIST = []
def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    #########################################################################
    R_deen,R_ende, SIMIL_en,SIMIL_de, img_embs, cap_embs = encode_data(
        model, val_loader, opt.log_step, logging.info)
    
    SIMIL_EN_LIST.append(SIMIL_en)
    SIMIL_DE_LIST.append(SIMIL_de)

    logging.info("(en-de) pair alignment ratio: %.4f" %(SIMIL_en))
    logging.info("(de-en) pair alignment ratio: %.4f" %(SIMIL_de))
    #########################################################################
    ############################ I2T and T2I for English #######################
    # caption retrieval
    (r1_en, r5_en, r10_en, medr_en, meanr_en),(r1_de, r5_de, r10_de, medr_de, meanr_de) = i2t(img_embs, cap_embs, measure=opt.measure)
    logging.info("(English)Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1_en, r5_en, r10_en, medr_en, meanr_en))

    logging.info("(German)Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1_de, r5_de, r10_de, medr_de, meanr_de))
    # image retrieval
    (r1i_en, r5i_en, r10i_en, medri_en, meanri_en),(r1i_de, r5i_de, r10i_de, medri_de, meanri_de) = t2i(img_embs, cap_embs, measure=opt.measure)
    logging.info("(English)Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i_en, r5i_en, r10i_en, medri_en, meanri_en))
    logging.info("(German)Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i_de, r5i_de, r10i_de, medri_de, meanri_de))
    # sum of recalls to be used for early stopping
    currscore_en = r1_en + r5_en + r10_en + r1i_en + r5i_en + r10i_en
    currscore_de = r1_de + r5_de + r10_de + r1i_de + r5i_de + r10i_de

    # record metrics in tensorboard
    tb_logger.log_value('SIMIL_en', SIMIL_en, step=model.Eiters)
    tb_logger.log_value('SIMIL_de', SIMIL_de, step=model.Eiters)
    tb_logger.log_value('r1_en', r1_en, step=model.Eiters)
    tb_logger.log_value('r5_en', r5_en, step=model.Eiters)
    tb_logger.log_value('r10_en', r10_en, step=model.Eiters)
    tb_logger.log_value('medr_en', medr_en, step=model.Eiters)
    tb_logger.log_value('meanr_en', meanr_en, step=model.Eiters)
    tb_logger.log_value('r1i_en', r1i_en, step=model.Eiters)
    tb_logger.log_value('r5i_en', r5i_en, step=model.Eiters)
    tb_logger.log_value('r10i_en', r10i_en, step=model.Eiters)
    tb_logger.log_value('medri_en', medri_en, step=model.Eiters)
    tb_logger.log_value('meanri_en', meanri_en, step=model.Eiters)
    tb_logger.log_value('rsum_en', currscore_en, step=model.Eiters)
    ###############################################################################
    # record metrics in tensorboard
    tb_logger.log_value('r1_de', r1_de, step=model.Eiters)
    tb_logger.log_value('r5_de', r5_de, step=model.Eiters)
    tb_logger.log_value('r10_de', r10_de, step=model.Eiters)
    tb_logger.log_value('medr_de', medr_de, step=model.Eiters)
    tb_logger.log_value('meanr_de', meanr_de, step=model.Eiters)
    tb_logger.log_value('r1i_de', r1i_de, step=model.Eiters)
    tb_logger.log_value('r5i_de', r5i_de, step=model.Eiters)
    tb_logger.log_value('r10i_de', r10i_de, step=model.Eiters)
    tb_logger.log_value('medri_de', medri_de, step=model.Eiters)
    tb_logger.log_value('meanri_de', meanri_de, step=model.Eiters)
    tb_logger.log_value('rsum_de', currscore_de, step=model.Eiters)
    ###############################################################################

    return currscore_en, currscore_de,SIMIL_en,SIMIL_de,R_deen,R_ende


def save_checkpoint(state, R_deen, R_ende, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
	np.save('R_deen.npy',R_deen.cpu().numpy())
	np.save('R_ende.npy',R_ende.cpu().numpy())


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

