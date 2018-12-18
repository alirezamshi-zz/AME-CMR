# Project 2 Machine Learning :
## Twittly supervised multi-lingual text classifier :
Alireza Mohammadshahi, Mohammadreza Banaei  

This projects aims to classify an unlabeled dataset (different social media sources in Switzerland) into three classes: 
1. Soccer, 2. Ice-hockey, 3. None.  

We have described the model completely in the report, but we can explain it briefly here.  
The dataset contains three different social media sources ( Twitter, Instagram, Web ) of people in Switzerland. Since the size of several texts are huge, we filter the texts that have tokens lower than 50. Then, we tokenize the corpus with a tokenizer that LSIR gives us that is usefull for Twitter datasets. Here is the example of a raw and tokenized version of a sample text:  
 ................................  
 ................................  
 
 
