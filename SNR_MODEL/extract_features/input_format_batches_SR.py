#run with python3

import json

# this script was used to extract the relevant annotated SRs from the files 'trainAfterRules_26_07_2021_cor_READY', 'valAfterRules_26_07_2021_cor_READY','testAfterRules_26_07_2021_cor_READY' that had all the annotated concepts. So that an input file for the SR related models could be used as input (defined in experiments.conf file)

ner_labels_first_level = 'DEF,OBLIG,RIGHT,LEFFECT,INTRO,IGNORE'

labels = ["DEFINIENDUM","DEFINIENS","DEF-INCLUSION","DEF-EXCLUSION","DEF-EXEMPLUM","SCOPE","ACTION","CONDITION","CONCESSION","PURPOSE","EXPERIENCER","AGENT","PATIENT","OBJECT","THEME","EXCEPTION","EFFECT","RATIONALE","NEG"]


less100 = ["IGNORE", "REF", "URL", "SITUATION", "RESULT", "NE_","NE_ORG", "RATIONALE",  "NE_OFFICE", "NE_PERSON","TIME_", "TIME_DATE_ABS", "TIME_DATE_REL_ENUNC", "TIME_FREQ", "DEF-EXCLUSION", "DEF-EXEMPLUM", "AGENT", "PATIENT", "OBJECT" ]


with_norms_flag = False
files = ['trainAfterRules_26_07_2021_cor_READY', 'valAfterRules_26_07_2021_cor_READY','testAfterRules_26_07_2021_cor_READY']

            
punctuation = [".",":","!","?",";",","]


for file_name in files:
    new_corpus = open(file_name + '.jsonl','r')
    lines = new_corpus.readlines()
    sentences = [json.loads(jline) for jline in lines]
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
            if ner[2] not in less100 and ner[2] in labels:
                TAG += 1
                sent[ner[0]] += ' ' + str(TAG) + 'I__' +  ner[2]
                sent[ner[1]] += ' ' + str(TAG) + 'F__' +  ner[2]
        if with_norms_flag:
            for ner in ners_sent:
                if ner[2] in ['DEF','OBLIG','RIGHT','LEFFECT','INTRO']:
                    if ner[2] == 'OBLIG':
                        ner[2] = 'OBRIG'
                    elif ner[2] == 'RIGHT':
                        ner[2] = 'DIREITO'
                    elif ner[2] == 'LEFFECT':
                        ner[2] == 'EFEITO'
                    print(sentence["id"])
                    print(ner)
                    sent[ner[0]] = 'I' + ner[2] + ' ' + sent[ner[0]]
                    sent[ner[1]] += ' F' + ner[2]

        fin_sent = []
        print(sent)
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
        print(final_sent)
        
            
    if  with_norms_flag:
        file_name_model = file_name + '_READY_SR.jsonl'
    else:
        file_name_model = file_name + '_READY_SR_simple.jsonl'

    ex = open(file_name_model,'w')
    print(file_name_model)
    print(len(final_ids))
    for index in range(0,len(final_ids)):
        json.dump({"id": final_ids[index], "ners": final_ners[index],  "sentence": real_final_sents[index]}, ex)
        ex.write("\n")
    ex.close()
                        
                                
                
                
                    
                        
                    

            
                    


