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
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  config = util.initialize_from_env()

  model = biaffine_ner_model.BiaffineNERModel(config)

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  session_config.allow_soft_placement = True
  with tf.Session(config=session_config) as session:
    model.restore(session)

    summary = model.predict(session)

