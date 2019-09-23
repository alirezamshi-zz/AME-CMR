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

### Building Word Embeddings :

You should first download the [aligned word vectors of fasttext](https://fasttext.cc/docs/en/aligned-vectors.html), and align them with bilingual lexicons.  
Then, run the `vocab.py` file to build the initial dictionary of the model.

```
vocab.py [-h] [--data_path DATA_PATH] [--model_path_en MODEL_PATH_EN]
                [--model_path_de MODEL_PATH_DE] [--data_name DATA_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
  --model_path_en MODEL_PATH_EN
  --model_path_de MODEL_PATH_DE
  --data_name DATA_NAME
                        {coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k
```

### Building the image embeddings:

You can download the precomputed embeddings of images from [here](https://github.com/ryankiros/visual-semantic-embedding/) and [here](https://github.com/ivendrov/order-embedding). Also, you can download the original images from here, then compute the image features on your image encoder.

### Run the main code : 

Now, you can train the model with `train.py` file : 

```
train.py [-h] [--data_path DATA_PATH] [--data_name DATA_NAME]
                [--vocab_path VOCAB_PATH] [--margin MARGIN]
                [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                [--word_dim WORD_DIM] [--embed_size EMBED_SIZE]
                [--grad_clip GRAD_CLIP] [--crop_size CROP_SIZE]
                [--num_layers NUM_LAYERS] [--learning_rate LEARNING_RATE]
                [--lr_update LR_UPDATE] [--workers WORKERS]
                [--log_step LOG_STEP] [--val_step VAL_STEP]
                [--logger_name LOGGER_NAME] [--resume PATH] [--max_violation]
                [--img_dim IMG_DIM] [--finetune] [--cnn_type CNN_TYPE]
                [--use_restval] [--measure MEASURE] [--use_abs] [--no_imgnorm]
                [--reset_train] [--dictionary DICTIONARY]
                [--WORDS_en WORDS_EN] [--WORDS_de WORDS_DE]
                [--WORDS_known_en WORDS_KNOWN_EN]
                [--WORDS_known_de WORDS_KNOWN_DE]
                [--train_set_ende TRAIN_SET_ENDE]
                [--test_set_ende TEST_SET_ENDE]
                [--train_set_deen TRAIN_SET_DEEN]
                [--test_set_deen TEST_SET_DEEN] [--maxup MAXUP]
                [--maxneg MAXNEG] [--knn KNN] [--const CONST]
                [--lr_translator LR_TRANSLATOR]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        path to datasets
  --data_name DATA_NAME
                        {coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k
  --vocab_path VOCAB_PATH
                        Path to saved vocabulary pickle files.
  --margin MARGIN       Rank loss margin.
  --num_epochs NUM_EPOCHS
                        Number of training epochs.
  --batch_size BATCH_SIZE
                        Size of a training mini-batch.
  --word_dim WORD_DIM   Dimensionality of the word embedding.
  --embed_size EMBED_SIZE
                        Dimensionality of the joint embedding.
  --grad_clip GRAD_CLIP
                        Gradient clipping threshold.
  --crop_size CROP_SIZE
                        Size of an image crop as the CNN input.
  --num_layers NUM_LAYERS
                        Number of GRU layers.
  --learning_rate LEARNING_RATE
                        Initial learning rate.
  --lr_update LR_UPDATE
                        Number of epochs to update the learning rate.
  --workers WORKERS     Number of data loader workers.
  --log_step LOG_STEP   Number of steps to print and record the log.
  --val_step VAL_STEP   Number of steps to run validation.
  --logger_name LOGGER_NAME
                        Path to save the model and Tensorboard log.
  --resume PATH         path to latest checkpoint (default: none)
  --max_violation       Use max instead of sum in the rank loss.
  --img_dim IMG_DIM     Dimensionality of the image embedding.
  --finetune            Fine-tune the image encoder.
  --cnn_type CNN_TYPE   The CNN used for image encoder (e.g. vgg19, resnet152)
  --use_restval         Use the restval data for training on MSCOCO.
  --measure MEASURE     Similarity measure used (cosine|order)
  --use_abs             Take the absolute value of embedding vectors.
  --no_imgnorm          Do not normalize the image embeddings.
  --reset_train         Ensure the training is always done in train mode (Not
                        recommended).
  --dictionary DICTIONARY
                        Path to dictionary
  --WORDS_en WORDS_EN   Path to words of dictionary
  --WORDS_de WORDS_DE   Path to words of dictionary
  --WORDS_known_en WORDS_KNOWN_EN
                        Path to words of dictionary
  --WORDS_known_de WORDS_KNOWN_DE
                        Path to words of dictionary
  --train_set_ende TRAIN_SET_ENDE
                        path to train set of (en-de) pairs
  --test_set_ende TEST_SET_ENDE
                        Path to test set of (en-de) pairs
  --train_set_deen TRAIN_SET_DEEN
                        Path to train set of (de-en) pairs
  --test_set_deen TEST_SET_DEEN
                        Path to test set of (de-en) pairs
  --maxup MAXUP         Maximum number of training examples
  --maxneg MAXNEG       Maximum number of negatives for the Extended RCSLS
  --knn KNN             Number of nearest neighbour for RCSLS
  --const CONST         Const to multiply in loss calculation
  --lr_translator LR_TRANSLATOR
                        Learning rate for translator
```

## Reference :

If you found this code useful, please cite the main paper.

## TO DO :

1. Add the pretrained-models
2. Add some results from paper
