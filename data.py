import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
#import nltk
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod

NO_CAP = 10


def get_paths(path, name='f30k', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids

###########################################################################################################
## Since I use precomp data, there is no need to change
###########################################################################################################
class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, vocab, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return len(self.ids)
###############################################################################################################
###############################################################################################################

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k
    """
##### maybe I should change vocab too ######################################################################
    def __init__(self, data_path, data_split, vocab):
        #################################################
        self.vocab = vocab
        #################################################
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())
##############################################################################################################
##############################################################################################################

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        ############################ assume the size of captions are the same #################################
        self.lengths = len(self.captions)
        #######################################################################################################
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.lengths:
            self.im_div = NO_CAP
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 10000
    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index/self.im_div
        image = torch.Tensor(self.images[img_id])
    ####################################################################################################
        caption = self.captions[index]
    ##########################################################
        vocab = self.vocab
    ####################################################
    #####################################################################################################
        # Convert caption (string) to word ids.
        tokens = str(caption).decode('utf-8').split()
	#tokens1 = nltk.tokenize.word_tokenize(str(caption).lower().decode('utf-8'))
	#print(tokens)
	#print(tokens1)
        caption = []
        if(index % 10 < 5):
            kind = 1
        elif(index % 10 >= 5):
            kind = 2
        caption.append(vocab('<start>',kind))
        caption.extend([vocab(token,kind) for token in tokens])
        caption.append(vocab('<end>',kind))
        target = torch.Tensor(caption)
    ################################################################z
        return image, target, index, img_id

    def __len__(self):
        #######################################################
        return self.lengths


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    ############################### Sorts by eng caption######################################
    ############################ add captions_de #############################################
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    ################################## Merge Captions for German caps##########################
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    Max = max(lengths)
    targets = torch.zeros(len(captions), Max).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        if(targets[i,end-1] >= 8194):
            targets[i,end:Max] = 8194
    ######################################################################
    return images, targets, lengths, ids

###################### Since I use precomp data, no need for this###############################
def get_loader_single(data_name, split, root, json, vocab, transform,
                      batch_size=100, shuffle=True,
                      num_workers=2, ids=None, collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset.
    if 'coco' in data_name:
        # COCO custom dataset
        dataset = CocoDataset(root=root,
                              json=json,
                              vocab=vocab,
                              transform=transform, ids=ids)
"""
    if'f30k' in data_name:
        dataset = FlickrDataset(root=root,
                                split=split,
                                json=json,
                                vocab=vocab,
                                transform=transform)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
##################################################################################################


################################ change input vocab --> vocab_en and vocab_de
def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader

## No chnage
def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform

##################################### Input Change ######################################################
def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith('_precomp'):
        train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                          batch_size, True, workers)
        val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                        batch_size, False, workers)
    ################### Since I use precomp data, no need for this ###########################
    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, 'train', opt)
        train_loader = get_loader_single(opt.data_name, 'train',
                                         roots['train']['img'],
                                         roots['train']['cap'],
                                         vocab, transform, ids=ids['train'],
                                         batch_size=batch_size, shuffle=True,
                                         num_workers=workers,
                                         collate_fn=collate_fn)

        transform = get_transform(data_name, 'val', opt)
        val_loader = get_loader_single(opt.data_name, 'val',
                                       roots['val']['img'],
                                       roots['val']['cap'],
                                       vocab, transform, ids=ids['val'],
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=collate_fn)
    ###########################################################################################
    return train_loader, val_loader

##################### change input vocab---> vocab_en and vocab_de
def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith('_precomp'):
        test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                         batch_size, False, workers)
    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, split_name, opt)
        test_loader = get_loader_single(opt.data_name, split_name,
                                        roots[split_name]['img'],
                                        roots[split_name]['cap'],
                                        vocab, transform, ids=ids[split_name],
                                        batch_size=batch_size, shuffle=False,
                                        num_workers=workers,
                                        collate_fn=collate_fn)

    return test_loader

