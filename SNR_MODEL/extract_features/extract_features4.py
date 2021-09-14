import torch
import json
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads
from transformers import BertModel, AutoModel, BertConfig  # or BertModel, for BERT without pretraining heads
import h5py
import numpy as np
import os
import itertools
import datetime
import sys

print(datetime.datetime.now())
device = "cuda:0" if torch.cuda.is_available() else "cpu" #can change to other gpus, if no gpu is found cpu will be used instead

if  sys.argv[1] == 'base': #if it is the baseline approch create embeedings for sentences without Norms
    files = ["trainAfterRules_26_07_2021_old_READY_NORMS.jsonl","valAfterRules_26_07_2021_old_READY_NORMS.jsonl", "testAfterRules_26_07_2021_old_READY_NORMS.jsonl"]
    output_emb = 'bert_hugfterRules_26_07_NoNorms.hdf5'
elif sys.argv[1] == 'not_base_train': #if it is not the baseline approch, but still training, create embeedings for sentences with gold Norms
    files = ["trainAfterRules_26_07_2021_cor_READY_READY_SR.jsonl","valAfterRules_26_07_2021_cor_READY_READY_SR.jsonl"]#,"testAfterRules_16_07_2021_cor_READY_READY_SR.jsonl"]
    output_emb = 'bert_hugfterRules_26_07_WithNorms_cor.hdf5'
    
elif sys.argv[1] == 'predict': #to make predictions for baseline, after generation is done, need to change "lm_path" in  experiments.conf file
    files = ["input_set.jsonl"]
    output_emb = 'input_set.hdf5'
    
#elif sys.argv[1] == 'predict': #to make predictions for other approaches, after generation is done, need to change "lm_path" in  experiments.conf file
#    files = ["input_set_withNorms.jsonl"]  --> file that has the segments with the norms predictions, so that NE and SR can be predicted as well
#    output_emb = 'input_set_withNorms.hdf5' --> need to change "lm_path" in  experiments.conf file for the corresponding models

else: #if it is not the baseline approch, add embeedings for sentences with predicted Norms
    files = ["testAfterRules_26_07_2021_cor_READY_READY_SR.jsonl"]
    output_emb = 'bert_hugfterRules_26_07_WithNorms_cor.hdf5'
    
sentences = {}


for file_name in files:
    file = open(file_name,'r')
    lines = file.readlines()
    sents = [json.loads(jline) for jline in lines]
    file.close()

    for sent in sents:
        sentences[sent["id"]] = sent["sentence"]


config = BertConfig.from_pretrained('neuralmind/bert-large-portuguese-cased', output_hidden_states=True)
model = BertModel.from_pretrained('neuralmind/bert-large-portuguese-cased',config=config)
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased',  do_lower_case=False)
model = model.to(device)


if 'WithNorms' not in output_emb:
    if os.path.exists(output_emb):
        os.remove(output_emb)
    writer = h5py.File(output_emb, 'w')
else:
    if os.path.exists(output_emb):
        writer = h5py.File(output_emb, 'a')
    else:
        writer = h5py.File(output_emb, 'w')
    
sentence_index = 0
for sent_id in sentences.keys():
    sent_tokens = sentences[sent_id]
    sent = ' '.join(sent_tokens)
    dataset_key ="{}".format(sent_id)

    sentence_index += 1

    input_id_by_word = [tokenizer.encode(x, add_special_tokens=False) for x in sent_tokens] # [ [ids do token1], [ids do token 2], ... ]
        
    total_ids = 0
    input_id_by_word_index = 0
    last_index = []
    input_batches = []
    for input_word in input_id_by_word: # [ids do token1]
        total_ids += len(input_word)
        if total_ids > 512:
            start = 0
            if last_index != []:
                start = last_index[-1]
            input_batches.append(input_id_by_word[start:input_id_by_word_index])
            last_index.append(input_id_by_word_index)
            total_ids = len(input_word)
        elif input_id_by_word_index == len(input_id_by_word) - 1:
            start = 0
            if last_index != []:
                start = last_index[-1]
            input_batches.append(input_id_by_word[start:])
        input_id_by_word_index += 1
        
    token_index = 0
    for input_batch in input_batches:  # [ [ids do token1], [ids do token 2], ... ] (512 max number of elements in list of lists)
        flat_inputs = list(itertools.chain.from_iterable(input_batch)) # [ ids do token1, ids do token 2, ... ]
        with torch.no_grad():
            outs = model(torch.as_tensor([flat_inputs]).to(device))
            hidden_states = outs[2]

            if dataset_key not in writer:
                writer.create_dataset(dataset_key, (len(sent_tokens), 1024, 4), dtype=np.float32)
            
            dset = writer[dataset_key]
            id_index = 0
            embedding_output = {}
            for id in flat_inputs:
                for j, layer_index in enumerate([-1,-2,-3,-4]):
                    if layer_index not in embedding_output.keys():
                        embedding_output[layer_index] = []
                    embedding_output[layer_index].append(hidden_states[layer_index][0][id_index])
                id_index += 1
                
            id_index = 0
            for word in input_batch: # [id1 id2 id3] ids of token x
                word_emb = {}
                for word_piece in word: # id1
                    for j, layer_index in enumerate([-1,-2,-3,-4]):
                        if layer_index not in word_emb.keys():
                            word_emb[layer_index] = []
                        word_emb[layer_index].append(embedding_output[layer_index][id_index])
                    id_index += 1
                if word_emb != {}:
                    for j, layer_index in enumerate([-1,-2,-3,-4]):
                        emb = torch.stack(word_emb[layer_index]).mean(dim=0)
                        dset[token_index, :, j] = emb.cpu()

                token_index += 1

                        
writer.close()

print(datetime.datetime.now())

print('done embeddings')
data = h5py.File(output_emb, 'r')

#the code bellow just prints the embeddings, can be commented
for sent_id in data.keys() :
    print('##########Sent##########')
    print(sent_id)
    ds_data = data[sent_id] # returns HDF5 dataset object
    print(ds_data)
    print(ds_data.shape, ds_data.dtype)
    arr = data[sent_id][:] # adding [:] returns a numpy array
    print (arr.shape, arr.dtype)
    print (arr)
print(len(data.keys()))
file.close()




