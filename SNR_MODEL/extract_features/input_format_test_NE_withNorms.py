# Run with python3

import json
import os


less100 = ["IGNORE", "REF", "URL", "SITUATION", "RESULT", "NE_","NE_ORG", "RATIONALE",  "NE_OFFICE", "NE_PERSON","TIME_", "TIME_DATE_ABS", "TIME_DATE_REL_ENUNC", "TIME_FREQ", "DEF-EXCLUSION", "DEF-EXEMPLUM", "AGENT", "PATIENT" ]


NE_labels = ["LREF","TREF","NE_ADM","TIME_DATE_REL_TEXT","TIME_DURATION"]

punctuation = [".",":","!","?",";",","]


new_corpus = open('testAfterRules_26_07_2021_cor_READY.jsonl','r') #this files correspond to the annotations of the test set
lines = new_corpus.readlines()
sentences = [json.loads(jline) for jline in lines]
new_corpus.close()

test_ners = open("../Norms_pred_test_ners.jsonl","r")  # This file that contains norms model predictions to be added into a segment
lines = test_ners.read()
true_ners = json.loads(lines)["ners"] # Get predicted NORMS
test_ners.close()

final_ners = []
real_final_sents = []
final_ids = []
TAG = 0
for sentence in sentences:
    ners_sent = sentence["ners"]
    sent = sentence["sentence"]
    final_ids.append(sentence["id"])

    # The following for is responsible for adding NE  gold labels into a segment, so that when adding the Norms predicted labels, a new mapping of the NE gold spans indexes is done (regarding the segment with the norms information)
    for ner in ners_sent:
        if ner[2] not in less100 and ner[2] in NE_labels:
            TAG += 1
            sent[ner[0]] += ' ' + str(TAG) + 'I__' +  ner[2]
            sent[ner[1]] += ' ' + str(TAG) + 'F__' +  ner[2]


    id_ners = []
    if sentence["id"] in true_ners.keys():
        id_ners = true_ners[sentence["id"]]
    
    # The following for is responsible for adding NORM predicted labels into a segment
    for ner in id_ners:
        if ner[2] in ['DEF','OBLIG','RIGHT','LEFFECT','INTRO']:
            if ner[2] == 'OBLIG':
                ner[2] = 'OBRIG'
            elif ner[2] == 'RIGHT':
                ner[2] = 'DIREITO'
            elif ner[2] == 'LEFFECT':
                ner[2] == 'EFEITO'
            sent[ner[0]] = 'I' + ner[2] + ' ' + sent[ner[0]]
            sent[ner[1]] += ' F' + ner[2]

    # Saving final segment tokens. Example: ['IOBRIG' , '2', '-', 'Organização', '1I__NE_ADM', ... 'mundial', '1F__NE_ADM', ..., 'FOBRIG']
    fin_sent = []
    for token in sent:
        fin_sent += token.split(' ')

    labels_ners = {}
    token_index = -1
    final_sent = []
    for token in fin_sent:
        if 'I__' in token or 'F__' in token: # if token is NE label
            token_label = token.split('__')[1] # get label from "ILABEL"/"FLABEL" string
            if 'I__' in token: # if token is NE start label, save label start  position
                labels_ners[token.replace('I__','')] = [token_index,'-',token_label]
            if 'F__' in token: # if token is NE end label, save label end position
                labels_ners[token.replace('F__','')][1] = token_index
        elif token != '' and token != ' ': #if normal token (not NE label), update token position and add token to final sent
            final_sent.append(token)
            token_index += 1
            
    sent_n = []
    for key in labels_ners.keys():
        sent_n.append(labels_ners[key])
        
    final_ners.append(sent_n)
    real_final_sents.append(final_sent)

    
if os.path.exists("testAfterRules_26_07_2021_cor_READY_READY_NE.jsonl"):
    os.remove("testAfterRules_26_07_2021_cor_READY_READY_NE.jsonl")
ex = open('testAfterRules_26_07_2021_cor_READY_READY_NE.jsonl','w')
print('testAfterRules_26_07_2021_cor_READY_READY_NE')
print(len(final_ids))

for index in range(0,len(final_ids)):
    json.dump({"id": final_ids[index], "ners": final_ners[index],  "sentence": real_final_sents[index]}, ex)
    ex.write("\n")
ex.close()
                    
                                
                
                
                    
                        
                    

            
                    


