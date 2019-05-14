# Multilingual Multimodal Embeddings with Adapting Aligned Languages :
## Code for building the bilingual lexicons:

This code is implemented in order to adjust ground-truth bilingual lexicons of MUSE: Multilingual Unsupervised and Supervised Embeddings to our datasets (Multi30k and MS-COCO).  

## Get initial bilingual lexicons of MUSE:  
To download lexicons for English, German and Japanese languages, you can simply run:  
### Multi30K:  
`mkdir en-de`
`cd en-de`
`wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-de.txt`
`wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-de.0-5000.txt`
`wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-de.5000-6500.txt`
`mkdir de-en`
`cd de-en`
`wget https://dl.fbaipublicfiles.com/arrival/dictionaries/de-en.txt`
`wget https://dl.fbaipublicfiles.com/arrival/dictionaries/de-en.0-5000.txt`
`wget https://dl.fbaipublicfiles.com/arrival/dictionaries/de-en.5000-6500.txt`
### MS-COCO:
`mkdir en-ja`
`cd en-ja`
`wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ja.txt`
`wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ja.0-5000.txt`
`wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ja.5000-6500.txt`
`mkdir ja-en`
`cd ja-en`
`wget https://dl.fbaipublicfiles.com/arrival/dictionaries/ja-en.txt`
`wget https://dl.fbaipublicfiles.com/arrival/dictionaries/ja-en.0-5000.txt`
`wget https://dl.fbaipublicfiles.com/arrival/dictionaries/ja-en.5000-6500.txt`
