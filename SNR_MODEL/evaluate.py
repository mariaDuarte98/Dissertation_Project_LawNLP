#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()

import util,biaffine_ner_model
import datetime
import json
import pandas

def save_csv(entities, id, name, args=None):
    if not os.path.isfile("./results" + name + ".csv"):
        data_frame = pandas.DataFrame([entities],
                                      columns=['Gold','Pred'],
                                      index=[id])
        data_frame.to_csv("./results" + name + ".csv")
    else:
        data_frame = pandas.read_csv("./results" + name + ".csv", index_col=0)
        result = []
        data_frame.loc[id] = entities
        data_frame.to_csv("./results" + name + ".csv")

    return 0

print(datetime.datetime.now())
if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  config = util.initialize_from_env()

  config['eval_path'] = config['test_path']
  print(config['eval_path'])

  model = biaffine_ner_model.BiaffineNERModel(config)

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  session_config.allow_soft_placement = True
  with tf.Session(config=session_config) as session:
    
    model.restore(session)
    summary, m_f1, pre, rec, pred_ners, predicted_sentences, ners_maps, gold_ners, predicted_sentences_ids, loss, gold_labels = model.evaluate(session,is_final_test=True, evaluate_loss=True, eval_train_path=False)


  
  #labels = ["OBJECT", "DEFINIENS", "DEFINIENDUM", "TIME_DATE_REL_TEXT", "SCOPE", "EFFECT"]
  print('###########')
  sentence_labels = {}
  if "pt_law_10_NORMS" in config["log_dir"]:  #file to write predicted labels of NORMS model to be used in the 2 phased system for the SR and NE Model
    if os.path.exists("Norms_pred_test_ners.jsonl"):
        os.remove("Norms_pred_test_ners.jsonl")
    file = open("Norms_pred_test_ners.jsonl","w")
    
  #reading gold  labels and adding them on the sentence to later write all annotated sentences into table
  gold_final_results = {}
  sentence_batch = 0
  for sentences_ners in gold_ners:
    for sentence_ner in sentences_ners:
        sent = predicted_sentences[sentence_batch][sentence_ner[0]]
        sent_id = predicted_sentences_ids[sentence_batch][sentence_ner[0]]
        if sent_id not in gold_final_results:
            gold_final_results[sent_id] = sent[:]
            
        if sent_id not in sentence_labels:
            sentence_labels[sent_id] = {}
            sentence_labels[sent_id]['id'] = sent_id
            sentence_labels[sent_id]['sentence'] = sent
            sentence_labels[sent_id]['ners'] = []
        label = str(list(ners_maps.keys())[list(ners_maps.values()).index(sentence_ner[3])])
        gold_final_results[sent_id][sentence_ner[1]] += '[' + label + sent[sentence_ner[1]] + ']'
        gold_final_results[sent_id][sentence_ner[2]] += '[' + label + sent[sentence_ner[2]] + ']'
        
    sentence_batch += 1
    
  #reading predicted labels and adding them on the sentence to later write all annotated sentences into table
  pred_final_results = {}
  sentence_batch = 0
  ners_for_2phased = {} #dictionary to save the predicted ners that will be used for the Hybrid and Two Phased model
  for sentences_ners in pred_ners:
    for sentence_ner in sentences_ners:
        sent = predicted_sentences[sentence_batch][sentence_ner[0]]
        sent_id = predicted_sentences_ids[sentence_batch][sentence_ner[0]]
        
        if sent_id not in sentence_labels:
            sentence_labels[sent_id] = {}
            sentence_labels[sent_id]['id'] = sent_id
            sentence_labels[sent_id]['sentence'] = sent
            sentence_labels[sent_id]['ners'] = []
            
        if sent_id not in pred_final_results:
            pred_final_results[sent_id] = sent[:]
            
        label = str(list(ners_maps.keys())[list(ners_maps.values()).index(sentence_ner[3])])
        sentence_labels[sent_id]['ners'].append([sentence_ner[1],sentence_ner[2],label])
        pred_final_results[sent_id][sentence_ner[1]] += '[' + label + sent[sentence_ner[1]] + ']'
        pred_final_results[sent_id][sentence_ner[2]] += '[' + label + sent[sentence_ner[2]] + ']'
        
        if "pt_law_10_NORMS" in config["log_dir"]: #saving in dict predicted norms when evaluating Norms model
            if sent_id not in ners_for_2phased.keys():
                ners_for_2phased[sent_id] = []
            ners_for_2phased[sent_id].append([sentence_ner[1], sentence_ner[2], label])
    sentence_batch += 1
    
  if "pt_law_10_NORMS" in config["log_dir"]: #saving in file all predicted norms when evaluating Norms model
      json.dump({"ners": ners_for_2phased},file)
      file.close()

  results_jsonl = {}
  print('######write to json####')
  if os.path.exists("Model_pred_labels.jsonl"):
    file = open('Model_pred_labels.jsonl','r') #{12: [ner1, ner2,...], 234:[ner1, ...]...}
    lines = file.readlines()
    sentences = [json.loads(jline) for jline in lines]
    for sent in sentences:
          results_jsonl[sent["id"]] = sent
    file.close()
    os.remove("Model_pred_labels.jsonl")

  test_json = open("Model_pred_labels.jsonl","w")  #json file to save all predicted results
  if "withNorms" in config["log_dir"]: #adjust start and end of predicted spans considering tokens I_NORMLABEL and E_NORMLABEL were added into segment
      for id in sentence_labels.keys(): #for each sentence
        pos = 0
        count_norms_found = 0
        for token in sentence_labels[sent_id]["sentence"]: # for each token
            index = pos - count_norms_found
            if 'OBRIG' in token or  'RIGHT' in token or 'DEF' in token or 'INTRO' in token or 'LEFFECT' in token: #save index of added tokens
                if id == "717":
                    print(token)
                count_norms_found += 1
                for ner in sentence_labels[id]["ners"]:
                    if ner[0]> index: #if starts after the additional token, do start - 1 and end - 1  # IOBRIG SEM[CONCESSION] PREJUIZO ... anterior FOBRIG erd[CONCESSION]
                        ner[0] = ner[0] - 1
                        ner[1] = ner[1] - 1
                    elif ner[1] > index:
                        ner[1] = ner[1] - 1
                    elif ner[0] == index:
                        if token[0] == 'F':
                            ner[0] = ner[0] - 1
                            ner[1] = ner[1] - 1
                        else:
                            ner[1] = ner[1] - 1
                    elif ner[1] == index:
                        if token[0] == 'F':
                            ner[1] = ner[1] - 1

                        
                        
            pos += 1

        
        
  for id in sentence_labels.keys():
    if id  in results_jsonl.keys():
        new_ners = sentence_labels[id]["ners"]
        results_jsonl[id]["ners"] += new_ners
    else:
        results_jsonl[id] = sentence_labels[id]
    json.dump(results_jsonl[id], test_json)
    test_json.write("\n")
    
  for id in results_jsonl.keys():
    if id not in sentence_labels.keys():
        json.dump(results_jsonl[id], test_json)
        test_json.write("\n")
        
  print(results_jsonl["717"])
  test_json.close()
    
print('######write to cvs####')
#for each sentence join tokens (with labels) into a single string
for sent in pred_final_results:
    res = ''
    for token in pred_final_results[sent]:
        res += ' ' + token
    pred_final_results[sent] = res

for sent in gold_final_results:
    res = ''
    for token in gold_final_results[sent]:
        res += ' ' + token
    gold_final_results[sent] = res
    

# Write in a table in csv format, to open in excel
log_dir = config["log_dir"]

if os.path.exists("./results" + log_dir.split('/')[1] + ".csv"):
    os.remove("./results" + log_dir.split('/')[1] + ".csv")
for key in gold_final_results.keys():
    if key in pred_final_results:
        save_csv([gold_final_results[key], pred_final_results[key]],key, log_dir.split('/')[1])
    else:
        save_csv([gold_final_results[key], []],key, log_dir.split('/')[1])
        
for key in pred_final_results.keys():
    if key not in gold_final_results:
        save_csv([[], pred_final_results[key]],key, log_dir.split('/')[1])


#print(datetime.datetime.now())

