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

# Agreement between gold labels and predicted labels #
def agreement(golden_jsonl, results_jsonl): #gold_ners, pred_ners, sentences_ids, sentences):
    gold_sentences = {}
    pred_sentences = {}
    model = 0
    for id in golden_jsonl.keys():
        if id not in gold_sentences.keys():
            gold_sentences[id] = {}
        for ner in golden_jsonl[id]["ners"]:
            for index in range(ner[0],ner[1] + 1):
                if index not in gold_sentences[id].keys():
                    gold_sentences[id][index] = []
                gold_sentences[id][index].append(ner[2])
                
    for id in results_jsonl.keys():
        if id not in pred_sentences.keys():
            pred_sentences[id] = {}
        for ner in results_jsonl[id]["ners"]:
            for index in range(ner[0],ner[1] + 1):
                if index not in pred_sentences[id].keys():
                        pred_sentences[id][index] = []
                pred_sentences[id][index].append(ner[2])

    
    for sentId in gold_sentences.keys():
        for index in gold_sentences[sentId].keys():
            gold_sentences[sentId][index] = sorted(gold_sentences[sentId][index])
    

    for sentId in pred_sentences.keys():
        for index in pred_sentences[sentId].keys():
            pred_sentences[sentId][index] = sorted(pred_sentences[sentId][index])
    
    sentences_num = 0
    avg_score = 0
    for sent_id in gold_sentences.keys():
        sentences_num += 1
        sent_score = 0
        num_tokens = len(golden_jsonl[sent_id]["sentence"])

        if sent_id not in  pred_sentences.keys():
            pred_sentences[sent_id] = {}
        
        for i in range(0, num_tokens):
            if i not in gold_sentences[sent_id].keys():
                gold_sentences[sent_id][i] = []
            if i not in pred_sentences[sent_id].keys():
                pred_sentences[sent_id][i] = []
                      
            gold_sentences[sent_id][i] = sorted(gold_sentences[sent_id][i])
            pred_sentences[sent_id][i] = sorted(pred_sentences[sent_id][i])
            if gold_sentences[sent_id][i] == [] and pred_sentences[sent_id][i] == []:
                sent_score += 1
            #elif gold_sentences[model_count][sent_id][i] == pred_sentences[model_count][sent_id][i]:
            elif len(list(set(gold_sentences[sent_id][i]).intersection(pred_sentences[sent_id][i]))) > 0:
                sent_score += len(list(set(gold_sentences[sent_id][i]).intersection(pred_sentences[sent_id][i]))) / max(len(gold_sentences[sent_id][i]), len(pred_sentences[sent_id][i]))
                #sent_score += 1
            else:
                sent_score += 0
        sent_score = sent_score / num_tokens
        avg_score += sent_score
    avg_score /= sentences_num
    '''
    for model_index in gold_ners:
        batch_count = 0
        gold_sentences[model] = {}
        for batch in model_index:
            for ner in batch: # ner = (sentid, start, end, label)
                if sentences_ids[model][batch_count][ner[0]] not in gold_sentences[model].keys():
                    gold_sentences[model][sentences_ids[model][batch_count][ner[0]]] = {}
                for index in range(ner[1],ner[2] + 1):
                    if index not in gold_sentences[model][sentences_ids[model][batch_count][ner[0]]].keys():
                        gold_sentences[model][sentences_ids[model][batch_count][ner[0]]][index] = []
                    gold_sentences[model][sentences_ids[model][batch_count][ner[0]]][index].append(ner[3])
            batch_count += 1
        model += 1
    
        
    model = 0
    for model_index in pred_ners:
        batch_count = 0
        pred_sentences[model] = {}
        for batch in model_index:
            for ner in batch: # ner = (sentid, start, end, label)
                if sentences_ids[model][batch_count][ner[0]] not in pred_sentences[model].keys():
                    pred_sentences[model][sentences_ids[model][batch_count][ner[0]]] = {}
                for index in range(ner[1],ner[2] + 1):
                    if index not in pred_sentences[model][sentences_ids[model][batch_count][ner[0]]].keys():
                        pred_sentences[model][sentences_ids[model][batch_count][ner[0]]][index] = []
                    pred_sentences[model][sentences_ids[model][batch_count][ner[0]]][index].append(ner[3])
            batch_count += 1
        model += 1
    
    for model in gold_sentences.keys():
        for sentId in gold_sentences[model].keys():
            for index in gold_sentences[model][sentId].keys():
                gold_sentences[model][sentId][index] = sorted( gold_sentences[model][sentId][index])
    
    for model in pred_sentences.keys():
        for sentId in pred_sentences[model].keys():
            for index in pred_sentences[model][sentId].keys():
                    pred_sentences[model][sentId][index] = sorted( pred_sentences[model][sentId][index])
    
    avg_score = 0
    sentences_num = 0
    model_count = 0
    for model_index in sentences:
        batch_count = 0
        for batch in model_index:
            sent_count = 0
            for sentence in batch:
                sentences_num += 1
                sent_score = 0
                num_tokens = len(sentence)
                sent_id = sentences_ids[model_count][batch_count][sent_count]
                if sent_id not in  gold_sentences[model_count].keys():
                    gold_sentences[model_count][sent_id] = {}
                if sent_id not in  pred_sentences[model_count].keys():
                    pred_sentences[model_count][sent_id] = {}
                for i in range(0, num_tokens):
                    if i not in gold_sentences[model_count][sent_id].keys():
                        gold_sentences[model_count][sent_id][i] = []
                    if i not in pred_sentences[model_count][sent_id].keys():
                        pred_sentences[model_count][sent_id][i] = []
                      
                    gold_sentences[model_count][sent_id][i] = sorted(gold_sentences[model_count][sent_id][i])
                    pred_sentences[model_count][sent_id][i] = sorted(pred_sentences[model_count][sent_id][i])
                    if gold_sentences[model_count][sent_id][i] == [] and pred_sentences[model_count][sent_id][i] == []:
                        sent_score += 1
                    #elif gold_sentences[model_count][sent_id][i] == pred_sentences[model_count][sent_id][i]:
                    elif len(list(set(gold_sentences[model_count][sent_id][i]).intersection(pred_sentences[model_count][sent_id][i]))) > 0:
                        sent_score += len(list(set(gold_sentences[model_count][sent_id][i]).intersection(pred_sentences[model_count][sent_id][i]))) / max(len(gold_sentences[model_count][sent_id][i]), len(pred_sentences[model_count][sent_id][i]))
                        #sent_score += 1
                    else:
                        sent_score += 0
                sent_score = sent_score / num_tokens
                avg_score += sent_score
                sent_count += 1
            batch_count += 1
        model_count += 1
    avg_score /= sentences_num
    '''
    
    return avg_score
    
def save_csv(entities, id, name, args=None):
    if not os.path.isfile("./results" + name + ".csv"):
        data_frame = pandas.DataFrame([entities],
                                      columns=['Gold','Pred'],
                                      index=[id])
        data_frame.to_csv("./results" + name + ".csv")
    else:
        data_frame = pandas.read_csv("./results" + name + ".csv", index_col=0)
        result = []
       # for el in range(0,4):
        #    if len(entities) >= el + 1:
         #       result.append(entities[el])
          #  else:
           #     result.append('NONE')
        data_frame.loc[id] = entities
        data_frame.to_csv("./results" + name + ".csv")

    return 0


if __name__ == "__main__":
  #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  config = util.initialize_from_env()
  if os.path.exists("Model_pred_labels.jsonl"):
    os.remove("Model_pred_labels.jsonl")
  if "baseline" in config["log_dir"]:
      models = config['models']
      for model in models:
        print("Start of evaluation of the model " + model)
        os.system('python evaluate.py ' + model)
    
  else:
      models = config['models']
      phase1_models = models[0]
      pahse2_models = models[1]
      print("Start of evaluation first level models")
      for model in phase1_models:
        print("Start of evaluation of the model " + model)
        os.system('python evaluate.py ' + model)

      current_dir = os.getcwd()
      os.chdir(current_dir + '/extract_features')
      #Update sentences with the predicted norms calculated before
      for model in pahse2_models:
        if "NE" in model:
            os.system('python input_format_test_NE_withNorms.py')
        else: #"SR" in model
            os.system('python input_format_test_SR_withNorms.py')
    
      #Created embeddings for sentences with the predicted norms calculated before
      os.system('python extract_features4.py not_base_test')
     
      os.chdir(current_dir)
      print("Start of evaluation second level models")
      for model in pahse2_models:
        print("Start of evaluation of the model " + model)
        os.system('python evaluate.py ' + model)

  
  tp = 0
  fp = 0
  fn = 0
  #gold_ners = []
  #pred_ners = []
  #sentences_ids = []
  #sentences = []
  files = config['test_output']
  for file_name in files:
    print(file_name)
    file = open(file_name, "r")
    content = file.readlines()
    tp += int(content[0].split(",")[0])
    fp += int(content[0].split(",")[1])
    fn += int(content[0].split(",")[2])
    #gold_ners.append(eval(content[1]))
    #pred_ners.append(eval(content[2]))
    #sentences_ids.append(eval(content[3]))
    #sentences.append(eval(content[4]))
    #id_sent_dict = eval(content[4])
    
  results_jsonl = {}
  file = open('Model_pred_labels.jsonl','r') # each line has {id: 123, ners:[...], sentence: O ...} --> predicted results
  lines = file.readlines()
  sentences = [json.loads(jline) for jline in lines]
  for sent in sentences:
    results_jsonl[sent["id"]] = sent
  file.close()
  
  golden_jsonl = {}
  file = open('extract_features/testAfterRules_26_07_2021_cor_READY.jsonl','r') # each line has {id: 123, ners:[...], sentence: O ...} --> golden results
  lines = file.readlines()
  sentences = [json.loads(jline) for jline in lines]
  for sent in sentences:
    golden_jsonl[sent["id"]] = sent
  file.close()
  
    
  score = agreement(golden_jsonl, results_jsonl) #gold_ners, pred_ners, sentences_ids, id_sent_dict)
  m_r = 0 if tp == 0 else float(tp)/(tp+fn)
  m_p = 0 if tp == 0 else float(tp)/(tp+fp)
  m_f1 = 0 if m_p == 0 else 2.0*m_r*m_p/(m_r+m_p)
  print("Agreement between gold and predicted NERS: " + str(score))
  print("Mention F1: {:.2f}%".format(m_f1*100))
  print("Mention recall: {:.2f}%".format(m_r*100))
  print("Mention precision: {:.2f}%".format(m_p*100))

  
    
