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
  

  
  if "baseline" in config["log_dir"]:
      current_dir = os.getcwd()
      os.chdir(current_dir + '/extract_features')
      os.system('python extract_features4.py base')
      os.chdir(current_dir)
      models = config['models']
      for model in models:
        print("Start of training of the model " + model)
        os.system('python train.py ' + model)
    
  else:
      current_dir = os.getcwd()
      os.chdir(current_dir + '/extract_features')
      print("generating embeddings first level models")
      os.system('python extract_features4.py base') #for norms model
      print("generating embeddings second level models")
      os.system('python extract_features4.py not_base_train') #for 2 phase models
      os.chdir(current_dir)
      models = config['models']
      phase1_models = models[0]
      pahse2_models = models[1]
      print("Start of training first level models")
      for model in phase1_models:
        print("Start of training of the model " + model)
        os.system('python train.py ' + model)

      os.chdir(current_dir)
      print("Start of training second level models")
      for model in pahse2_models:
        print("Start of training of the model " + model)
        os.system('python train.py ' + model)

  

  
    
