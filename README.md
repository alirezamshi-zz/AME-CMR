# Multilingual Multimodal Embeddings with Adapting Aligned Languages :

Pytorch implementation of the paper: *Aligning Multilingual Word Embeddings for Cross-Modal Retrieval Task*, Alireza Mohammadshahi, Remi Lebret, Karl Aberer, 2019 (EMNLP-FEVER 2019)

## Dependencies : 
You should install the following packages for train/testing the model: 
- Python 2.7
- [Pytorch](https://pytorch.org/) > 0.2 
- [Numpy](https://numpy.org/)
- [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
- [Pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)
- [Torchvision](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)
- [EuroParl](https://www.statmt.org/europarl/)

## Training :

To train the image-caption retrieval model, you should first build the bilingual lexicons which are used for aligning the word embeddings.

### Code for building the bilingual lexicons:

This code is implemented in order to adjust ground-truth bilingual lexicons of [MUSE: Multilingual Unsupervised and Supervised Embeddings](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries) to our datasets (Multi30k and MS-COCO).  

### Get initial bilingual lexicons of MUSE:  
To download lexicons for English, German and Japanese languages, you can simply run:  
#### Multi30K:  
```
mkdir en-de 
cd en-de  
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-de.txt  
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-de.0-5000.txt  
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-de.5000-6500.txt  
mkdir de-en  
cd de-en  
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/de-en.txt  
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/de-en.0-5000.txt  
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/de-en.5000-6500.txt
```
#### MS-COCO:
```
mkdir en-ja  
cd en-ja  
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ja.txt  
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ja.0-5000.txt  
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ja.5000-6500.txt  
mkdir ja-en  
cd ja-en  
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/ja-en.txt  
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/ja-en.0-5000.txt  
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/ja-en.5000-6500.txt  
```
### Main code for building lexicons:  
For building and adjusting bilingual lexicons, we first combine "full" and "test" set of MUSE benchmark, then we adjust it to our dataset, and use it for training. For the test set, we adjust "train" set of MUSE benckmark to our dataset.  
Run `adjust_lexicons.py` to get the desired lexicons for both datasets:  
```
adjust_lexicons.py [-h] [--en_de_dir EN_DE_DIR] [--de_en_dir DE_EN_DIR]
                          [--en_ja_dir EN_JA_DIR] [--ja_en_dir JA_EN_DIR]
                          [--en_multi30k EN_MULTI30K] [--en_mscoco EN_MSCOCO]
                          [--de_multi30k DE_MULTI30K] [--ja_mscoco JA_MSCOCO]
                          [--offset_multi30k OFFSET_MULTI30K]
                          [--offset_mscoco OFFSET_MSCOCO]

optional arguments:
  -h, --help            show this help message and exit
  --en_de_dir EN_DE_DIR
                        Path to (en-de) lexicons
  --de_en_dir DE_EN_DIR
                        Path to (de-en) lexicons
  --en_ja_dir EN_JA_DIR
                        Path to (en-ja) lexicons
  --ja_en_dir JA_EN_DIR
                        Path to (ja-en) lexicons
  --en_multi30k EN_MULTI30K
                        Path to English words of Multi30k dataset
  --en_mscoco EN_MSCOCO
                        Path to English words of MS-COCO dataset
  --de_multi30k DE_MULTI30K
                        Path to German words of Multi30k dataset
  --ja_mscoco JA_MSCOCO
                        Path to Japanese words of MS-COCO dataset
  --offset_multi30k OFFSET_MULTI30K
                        Divide English and German words (if you have one file
                        for words)
  --offset_mscoco OFFSET_MSCOCO
                        Divide English and Japanese words (if you have one
                        file for words)
```

Note: word files for both dataset should be in pickle format.
