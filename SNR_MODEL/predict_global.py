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


if __name__ == "__main__":
  config = util.initialize_from_env()
  if os.path.exists("Model_pred_labels.jsonl"):
    os.remove("Model_pred_labels.jsonl")
  if "baseline" in config["log_dir"]:
      models = config['models']
      for model in models:
        print("Start of prediction for the" + model)
        os.system('python predict.py ' + model)
    
  else:
      models = config['models']
      phase1_models = models[0]
      pahse2_models = models[1]
      print("Start of prediction for first level models")
      for model in phase1_models:
        print("Start of prediciton for the model " + model)
        os.system('python predict.py ' + model)

      current_dir = os.getcwd()
      os.chdir(current_dir + '/extract_features')
      #Update sentences with the predicted norms calculated before
      #the file in these scripts needs to be the corresponding value(predict_path) in the ocnfiguration of the models
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
        os.system('python predict.py ' + model)

  



  
    
