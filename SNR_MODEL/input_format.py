#run with python3

import json

ner_labels_first_level = 'DEF,OBLIG,RIGHT,LEFFECT,INTRO,IGNORE'

file_name = '2lv_100_sonia.jsonl'
new_corpus = open(file_name,'r')
lines = new_corpus.readlines()
sentences = [json.loads(jline) for jline in lines]
new_corpus.close()

sentences_tokens = []
sentences_ners = []

error = 0

    
for sentence in sentences:
    sentence_token = []
    sentence_token_withtag = []
    for t in sentence["tokens"]:
        if t["text"] != "#":
            sentence_token.append(t["text"])
        sentence_token_withtag.append(t["text"])
    sentences_tokens.append(sentence_token)

    sentence_ners = []
    found_first_tag = False
    
    # 1lv_spans #
    for span in sentence["spans"]:
        # start of 1lv_span #
        if  found_first_tag == False:
            found_first_tag = True
            sentence_ners.append([span["token_start"], 0, span["label"]])
            # making sure the span is a 1lv_span
            if span["label"] not in ner_labels_first_level:
                error = 1
                "There is an annotation error!!!!"
        # end of 1lv_span #
        else:
            # making sure the start of 1lv_span matches the end #
            # and that the end corresponds to the end of the sentence #
            if span["label"] != sentence_ners[0][2] or  span["token_start"] != len(sentence_token_withtag) - 1:
                error = 1
                "There is an annotation error!!!!"
            elif sentence_token_withtag[-2] in [".",":","!","?"]:  #making sure the last punctuation of the sentence isn't included in the category (oblig, def...)
                sentence_ners[0][1] = len(sentence_token)  - 2
            else:
                sentence_ners[0][1] = len(sentence_token)  - 1
    #sentences_ners.append(sentence_ners)
    
    # 2lv_spans #
    for rel in sentence["relations"]:
        #when the relation does not use the last word
        # add - 1 to positions considering first # is no longer in sentence
        if rel["head"] != len(sentence_token_withtag) - 1 and rel["child"] != len(sentence_token_withtag) - 1:
            head = 1
            child = 1
            if rel["head"] < rel["child"]:
                if sentence_token_withtag[rel["head"]] in [".",":","!","?"]: #if it starts with a punctuation, not include that token on the entity (next one)
                    head = 0
                    print("found punctuation!!!")
                    error = 1
                if sentence_token_withtag[rel["child"]] in [".",":","!","?"]: #if it ends with a punctuation, not include that token on the entity (the one before)
                    child = 2
                    print("found punctuation!!!")
                    error = 1
                sentence_ners.append([rel["head"] - head, rel["child"] - child, rel["label"]])
            else:
                if sentence_token_withtag[rel["child"]] in [".",":","!","?"]: #if it starts with a punctuation, not include that token on the entity (next one)
                    child = 0
                    print("found punctuation!!!")
                    error = 1
                if sentence_token_withtag[rel["head"]] in [".",":","!","?"]: #if it ends with a punctuation, not include that token on the entity (the one before)
                    head = 2
                    print("found punctuation!!!")
                    error = 1
                sentence_ners.append([rel["child"] - child, rel["head"] - head, rel["label"]])
        # there is a sentence with a snr entity using the last #
        else:
            last = 1
            if sentence_token_withtag[-2] in [".",":","!","?"]:  #making sure the last punctuation of the sentence isn't included in the entity (lref ...)
                last = 2
                print("found punctuation!!!")
                error = 1
            if rel["head"] == len(sentence_token_withtag) - 1: # uses last # in the relation #
                sentence_ners.append([rel["child"] - 1, len(sentence_token)  - last, rel["label"]])
                
            elif rel["child"] == len(sentence_token_withtag) - 1:  # uses last # in the relation #
                sentence_ners.append([rel["head"] - 1, len(sentence_token)  - last, rel["label"]])
            
        
    
    sentences_ners.append(sentence_ners)

    # "relations":[{"head":1,"child":6,"head_span":{"start":2,"end":4,"token_start":1,"token_end":1,"label":null},"child_span":{"start":31,"end":43,"token_start":6,"token_end":6,"label":null},"color":"#b5c6c9","label":"THEME"}
    


json_examples = [{"doc_key": "batch_01", "ners": sentences_ners,  "sentences": sentences_tokens}]

file_name = 'json_examples.jsonl'
ex = open(file_name,'w')
ex.write(str(json_examples))
ex.close()


if error == 0:
    print("File is done, no errors found!")
        

