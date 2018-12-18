# Project 2 Machine Learning :
## Twittly supervised multi-lingual text classifier :
Alireza Mohammadshahi, Mohammadreza Banaei  

This projects aims to classify an unlabeled dataset (different social media sources in Switzerland) into three classes: 
1. Soccer, 2. Ice-hockey, 3. None.  

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
 
 ......................................
 
 ## Training Model :
 
 
 ## Evaluation :
 
 Here, you can run the `eval.py` file to reproduce the results with pre-train weights of the model.  
 The data is one ##### link ##############  
 
 
