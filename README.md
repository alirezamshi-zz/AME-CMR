# Project 2 Machine Learning :
## Twittly supervised multi-lingual text classifier :
Alireza Mohammadshahi, Mohammadreza Banaei  

This projects aims to classify an unlabeled dataset (different social media sources in Switzerland) into three classes: 
1. Soccer in Switzerland, 2. Ice-hockey in Switzerland, 3. None.  

We have described the model completely in the report, but we can explain it briefly here.  
The dataset contains three different social media sources ( Twitter, Instagram, Web ) of people in Switzerland. Since the size of several texts are huge, we filter the texts that have tokens lower than 50. 

## Requirement :
python 3  
numpy  
Pytorch 0.4  
fasttext 0.1  
## Tokenizer :  

We tokenize the corpus with a tokenizer that LSIR gives us that is usefull for Twitter datasets. Here is the example of a raw and tokenized version of a sample text:  

Sample 1:  

input : Less Monday - The Tuscany Session - Album Out Now #starchmusic #newmusic #swissmusic #funky #feelgood #lessmonday 📷by Anja Schori  
output : less monday - the tuscany session - album out now <hashtag> starchmusic <hashtag> newmusic <hashtag> swissmusic <hashtag> funky <hashtag> feelgood <hashtag> lessmonday 📷 by anja schori  

Sample 2:  
input : Pool with a view ⠀ . 🏊‍♂️⛰🏊‍♂️ ⠀ . Book a room in one of the highest villages in Europe with @adnaaffair and work on your #altitudetraining  
output : pool with a view ⠀ . 🏊 ‍ ♂ ️ ⛰ 🏊 ‍ ♂ ️ ⠀ . book a room in one of the highest villages in europe with <user> and work on your <hashtag> altitudetraining  
## Data Generation

Next, we use a file named `data_generation.py` which builds the data in each training process. It would add the labeled texts with texts that have initial labels, then randomly chooses unlabeled data from corpus with the same size of labeled data to balance the train/validation data.  
 
 Here is the detail of this file :  
 In this function, we build a training and validation data based on initial signs for each classes. So, we check each texts in the corpus to determine it's classes. If it doesn't assign to two first classes, we give it the sign of `None`.  
 Here is the initial signs for each classes:  
 ### Switerland Football sign:  
 `SIGN_F=['raiffeisen super league', 'valon fazliu', 'fc <allcaps> lugano', 'fc <allcaps> basel', 'fc <allcaps> gallen', 'fc <allcaps> zurich' , 'fc <allcaps> thun' , 'fc <allcaps> sion' , 'xamax' , 'ybfc' , 'fc <allcaps> luzern' , 'fc <allcaps> young boys' , "swiss cup" , "swiss super league" ]`
 ### Switzerland ice-hockey sign:  
 `SIGN_H=[" nhl " , "spengler cup" , "spenglercup" ,  "national league" , "ehc <allcaps> kloten" , "ehc <allcaps> winterthur" , "ehc <allcaps> visp" , "hc <allcaps> ajoie" , "hc <allcaps> thurgau" , "gck <allcaps> lions" , "evz <allcaps> academy" , "sc <allcaps> langenthal" , "ehc <allcaps> olten", "ev <allcaps> train" , "sc <allcaps> bern" , "lausanne hc <allcaps>" , "scl <allcaps> tigers" ,  "zsc <allcaps> linos" , "hc <allcaps> lugano" , "hc <allcaps> davos" ]`
 ### Non-Switzerland Football sign:  
 `SIGN_O=["man utd", "barça"  ,"barca" , "juventus", "man city", "manchester united" , "benfica" , "porto" , "spurs", " psg " , "neymar" , "messi" , "ronaldo" , "real madrid" , "bayern munich" ]`  
 ### Non-Switzerland ice-hockey:  
 `SIGN_OH=[" khl " , " nhl " , "stanley cup" , "stanleycup" , "jokerit" , "smashville" , "leijonat"]`
 
 You can run the `data_generation.py` for `tokenized_en.txt` file to see the output in `FH_dataset.txt`.  
 ## Training the word embedding:  
 
 For building our word embedding which is specific to the corpus, we use FastText skigram algorithms with `minn = 3` , `maxn = 6` and `dim = 300`. Here is the code:  
 `./fastext skipgram -input tokenized.txt -output out.txt --minn 3 --maxn 6 --dim 300`  
 
 Note: you should first install fasttext from the repository : https://github.com/facebookresearch/fastText  
 
 Here is the embedding vectors for languages that we train :  
 English:  ##############link #################  
 German: ############### link ################  
 French: ################## link ################  

 
 ## Training Model :
 In this section, we go to the details of functions in `run.py` file:  
 This function would do the training for each languages separately, and in each language, first the code would run `data_generation.py` file to build the train/val set. Next, it runs the `train_model.py` code for three ensembles. Then, we extract the last layer of GRU to get the probabilities for each classes. Here, we use active learning strategy. We pick texts which model detect with high probability, and ask the user ro check the annotation. The user can skip it by typing `100`. At the end of each training mode, we ask the user to do another loop or not.  
 Here is the detail of each function: 
   
 `train_model.py`:  
 Here the model loads the data which is generated with `data_generation.py` file, and save the best model in a `.pth` file. As mentioned in paper, we use embedding+GRU as our training model.  
 
 `extract_high_precision.py`:  
 Here we load the best model for each ensembles, and extract the probabilites in the last layer of model. So, for each text, we have a 3*1 vector which shows the probability of assigning it to a specific class.  
 
 ### Active Learning :  
 Here we change the probability threshold until a proper amount of ice-hockey and football texts are found ( these texts are the ones that are not in annotated file, and doesn't have any initial signs). So, we pick randomly 50 texts from them, and ask user to check the label for the model.  
 
 ## Evaluation :
 
 Here, you can run the `eval.py` file to reproduce the results with pre-train weights of the model.  
 For the test data, we choose randomly from corpus which contain three languages, and give them label manually. Here is the test data and it's label:  
 
### Tokenized data:  
English:  https://drive.google.com/open?id=1sfi2nZq_4sSgndma-ifKU8zW6Bv_x24c  
German:  https://drive.google.com/open?id=1YGasi6Os6iwdb8Dg3AKM71XL9V4naTPe  
French:  https://drive.google.com/open?id=1a734gB5E5SyHSXhlkj6LYwCua3VhgDtA  
### Labels:  
English:  https://drive.google.com/open?id=1sU5T30lKpC1-jHuHcjPtEmUd97BAbW7g  
German:  https://drive.google.com/open?id=1Qt05Z8GKKmy6ll8apnKCgtLR2DKtUaCw  
French:  https://drive.google.com/open?id=15pvBr7OkxFU3yXdWopRDp7b49OtWNk5p  
 
 In this function, we first separate the test set into three file based on language type of each text with FastText language identifier. Then, for the multi-lingual model, we pass each file to it's related model, and gather the result. For the baseline, we translate French and German texts to English, and give the whole test set to English model.  
 You can run the `eval.py` file to reproduce the result of these two models.  
 
 
 
