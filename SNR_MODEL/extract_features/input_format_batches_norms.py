#run with python3

import json


# this script was used to turn the each set of data ('trainAfterRules_26_07_2021_cor', 'valAfterRules_26_07_2021_cor', 'testAfterRules_26_07_2021_cor') from the prodigy annotation format to the input format of the Norms model (defined in experiments.conf file)


ner_labels_first_level = 'DEF,OBLIG,RIGHT,LEFFECT,INTRO,IGNORE'

files = ['trainAfterRules_26_07_2021_cor', 'valAfterRules_26_07_2021_cor', 'testAfterRules_26_07_2021_cor']
punctuation = [".",":","!","?",";",","]

for file_name in files:
    new_corpus = open(file_name + '.jsonl','r')
    lines = new_corpus.readlines()
    sentences = [json.loads(jline) for jline in lines]
    new_corpus.close()

    sentences_tokens = []
    sentences_token_withtag = []
    sentences_ners = []
    sentences_ids = []

    error = 0

        
    for sentence in sentences:
        sentence_token = []
        sentence_token_withtag = []
        for t in sentence["tokens"]:
            if t["text"] != "#":
                sentence_token.append(t["text"])
            sentence_token_withtag.append(t["text"])
        sentences_tokens.append(sentence_token)
        sentences_token_withtag.append(sentence_token_withtag)

        sentence_ners = []
        
        # 1lv_spans #
        if len(sentence["spans"]) == 2:
            found_first_tag = False
            for span in sentence["spans"]:
                if  found_first_tag == False:
                    found_first_tag = True
                    sentence_ners.append([span["token_start"], '-', span["label"]])
                    # making sure the span is a 1lv_span
                    if span["label"] not in ner_labels_first_level:
                        error = 1
                        print("There is an annotation error!!!!baddddddd")
                # end of 1lv_span #
                else:
                    if span["label"] != sentence_ners[0][2] or  span["token_start"] != len(sentence_token_withtag) - 1:
                        error = 1
                        if span["label"] != sentence_ners[0][2]:
                            print("There is an annotation errorreallly badddd!!!!")
                        if span["token_start"] != len(sentence_token_withtag) - 1:
                            if sentence_token_withtag[-2] in punctuation:  #making sure the last punctuation of the sentence isn't included in the category (oblig, def...)
                                sentence_ners[0][1] = span["token_start"]  - 2 # take out first and last # and punctuation
                            else:
                                sentence_ners[0][1] = span["token_start"]  - 1
                    elif sentence_token_withtag[-2] in punctuation:
                        sentence_ners[0][1] = span["token_start"]  - 3 # take out first and last # and punctuation
                    else:
                        sentence_ners[0][1] = span["token_start"]  - 2 # take out first and last #
        
        elif len(sentence["spans"]) % 2 == 0:
            print("more than 1 norm!!")
            span_index = 0
            sent_ners_labels = {}
            for span in sentence["spans"]:
                if span["label"] not in sent_ners_labels.keys(): # inicio da categoria
                    if sentence_token_withtag[span["token_start"]] == "#":
                        sentence_ners.append([span["token_start"], '-', span["label"]])
                        sent_ners_labels[span["label"]] = span_index
                        span_index += 1
                    else:
                        sentence_ners.append([span["token_start"] - 1, '-', span["label"]])
                        sent_ners_labels[span["label"]] = span_index
                        span_index += 1

                
                else:
                    index = sent_ners_labels[span["label"]]
                    if span["token_start"] == len(sentence_token_withtag) - 1: #anotated on last #
                        if sentence_token_withtag[-2] in punctuation:
                            sentence_ners[index][1] = span["token_start"]  - 3
                        else:
                            sentence_ners[index][1] = span["token_start"]  - 2
                        sent_ners_labels.pop(span["label"])  # found begining and end, done with the norm
                    
                    elif sentence_token_withtag[span["token_start"]] in punctuation:  #making sure the last punctuation of the sentence isn't included in the category (oblig, def...)
                        sentence_ners[index][1] = span["token_start"]  - 2
                        sent_ners_labels.pop(span["label"])  # found begining and end, done with the norm
                    else:
                        sentence_ners[index][1] = span["token_start"]  - 1
                        sent_ners_labels.pop(span["label"])  # found begining and end, done with the norm
        else:
            print("there is an error!!! odd number of spans")
            print(sentence["meta"]["segment_id"])
        
        sentences_ners.append(sentence_ners)
        sentences_ids.append(sentence["meta"]["segment_id"])

    '''size = 30
    file_name_model = file_name + '_' + str(size) + '_READY_NORMS.jsonl'
    ex = open(file_name_model,'w')
    for i in range(0, len(sentences_ners), size):
        json_examples = {"doc_key": "batch_" + str(batch_num), "ners": sentences_ners[i:i + size],  "sentences": sentences_tokens[i:i + size], "ids": sentences_ids[i:i + size]}
        json.dump(json_examples, ex)
        ex.write("\n")
        batch_num += 1
    ex.close()'''
    file_name_model = file_name + '_READY_NORMS.jsonl'
    ex = open(file_name_model,'w')
    print(file_name_model)
    print(len(sentences_ids))
    for index in range(0,len(sentences_ids)):
        json.dump({"id": sentences_ids[index], "ners": sentences_ners[index],"sentence": sentences_tokens[index]}, ex)
        ex.write("\n")
    ex.close()
        
    
    print(len(sentences_tokens))

    if error == 0:
        print("File is done, no errors found!")
        


