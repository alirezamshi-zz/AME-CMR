# Project 2 Machine Learning :
## Twittly supervised multi-lingual text classifier :
Alireza Mohammadshahi, Mohammadreza Banaei  

This projects aims to classify an unlabeled dataset (different social media sources in Switzerland) into three classes: 
1. Soccer in Switzerland, 2. Ice-hockey in Switzerland, 3. None.  

We have described the model completely in the report, but we can explain it briefly here.  
The dataset contains three different social media sources ( Twitter, Instagram, Web ) of people in Switzerland. Since the size of several texts are huge, we filter the texts that have tokens lower than 50. 

## Tokenizer :  

We tokenize the corpus with a tokenizer that LSIR gives us that is usefull for Twitter datasets. Here is the example of a raw and tokenized version of a sample text:  

English:  

 ................................  
 ................................  

French:  

.................................  
.................................  
 
 German:  
 
 ................................  
 ................................  
 
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
 
 
 ## Training Model :
 
 
 ## Evaluation :
 
 Here, you can run the `eval.py` file to reproduce the results with pre-train weights of the model.  
 The data is one ##### link ##############  
 
 
