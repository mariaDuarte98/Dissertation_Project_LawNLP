# Dissertation_Project_LawNLP
This folder contains the code for the SNR(Semantic Norm Recognition) system. 

The SNR system is an information extraction system to extract norms, named entities and semantic relationships from portuguese consumer laws. This system uses 3 models (Norms Model, NE Model, SR model, NE with Norms, SR with Norms) each one responsible for predicting a group of labels. Each one corresponds to a nested NER model, based on the model of Yu, Bohnet and Poesio (from https://github.com/juntaoy/biaffine-ner). The base code came from them, where some fodifications were done, since we only use BERT as embeddings and they used 3 different types.


