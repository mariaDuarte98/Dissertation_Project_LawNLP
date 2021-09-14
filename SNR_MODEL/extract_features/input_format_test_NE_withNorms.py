#run with python3

import json
import os
ner_labels_first_level = 'DEF,OBLIG,RIGHT,LEFFECT,INTRO,IGNORE'

punctuation = [".",":","!","?",";",","]

less100 = ["IGNORE", "REF", "URL", "SITUATION", "RESULT", "NE_","NE_ORG", "RATIONALE",  "NE_OFFICE", "NE_PERSON","TIME_", "TIME_DATE_ABS", "TIME_DATE_REL_ENUNC", "TIME_FREQ", "DEF-EXCLUSION", "DEF-EXEMPLUM", "AGENT", "PATIENT" ]


NE_labels = ["LREF","TREF","NE_ADM","TIME_DATE_REL_TEXT","TIME_DURATION"]


new_corpus = open('testAfterRules_26_07_2021_cor_READY.jsonl','r')
lines = new_corpus.readlines()
sentences = [json.loads(jline) for jline in lines]
#sentences = sentences_ents
new_corpus.close()
final_ners = []
real_final_sents = []
final_ids = []
TAG = 0
for sentence in sentences:
    ners_sent = sentence["ners"]
    sent = sentence["sentence"]
    final_ids.append(sentence["id"])

    for ner in ners_sent:
        if ner[2] not in less100 and ner[2] in NE_labels:
            TAG += 1
            sent[ner[0]] += ' ' + str(TAG) + 'I__' +  ner[2]
            sent[ner[1]] += ' ' + str(TAG) + 'F__' +  ner[2]

    test_ners = open("../Norms_pred_test_ners.jsonl","r")
    lines = test_ners.read()
    true_ners = json.loads(lines)["ners"]
    if sentence["id"] in true_ners.keys():
        id_ners = true_ners[sentence["id"]]
    else:
        id_ners = []
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

    fin_sent = []
    for token in sent:
        fin_sent += token.split(' ')

    labels_ners = {}
    token_index = -1
    final_sent = []
    for token in fin_sent:
        if 'I__' in token or 'F__' in token:
            token_label = token.split('__')[1]
            if token_label not in labels_ners.keys() and 'I__' in token:
                labels_ners[token.replace('I__','')] = []
            if 'I__' in token:
                labels_ners[token.replace('I__','')].append([token_index,'-',token_label])
            if 'F__' in token:
                not_found = True
                il = - 1
                while not_found:
                    if labels_ners[token.replace('F__','')][il][1] == '-':
                        labels_ners[token.replace('F__','')][il][1] = token_index
                        not_found = False
                    else:
                        il -= 1
        elif token != '' and token != ' ':
            final_sent.append(token)
            token_index += 1
    sent_n = []
    for key in labels_ners.keys():
        if labels_ners[key] != []:
            sent_n += labels_ners[key]
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
                    
                                
                
                
                    
                        
                    

            
                    


