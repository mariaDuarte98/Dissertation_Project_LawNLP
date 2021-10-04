#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, io
import errno
import time
import pandas

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import util,biaffine_ner_model
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
import pyhocon
import sys
#import datetime

def mkdirs(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
  return path


def save_csv(entities, epoch, name, args=None):
    if not os.path.isfile("./" + name + ".csv"):
        data_frame = pandas.DataFrame([entities],
                                      columns=['Loss'],
                                      index=[id])
        data_frame.to_csv("./" + name + ".csv")
    else:
        data_frame = pandas.read_csv("./" + name + ".csv", index_col=0)
        result = []
        data_frame.loc[epoch] = entities
        data_frame.to_csv("./" + name + ".csv")

    return 0
    
def save_csv_eval(entities, epoch, name, args=None):
    if not os.path.isfile("./" + name + ".csv"):
        data_frame = pandas.DataFrame([entities],
                                      columns=['F1', 'Precision' ,'Recall'],
                                      index=[id])
        data_frame.to_csv("./" + name + ".csv")
    else:
        data_frame = pandas.read_csv("./" + name + ".csv", index_col=0)
        result = []
        data_frame.loc[epoch] = entities
        data_frame.to_csv("./" + name + ".csv")

    return 0

#print(datetime.datetime.now())

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"


  dim_learning_rate = Real(low=1e-3, high=1e-1, prior='log-uniform',name='learning_rate')
  #dim_ffnn_size = Integer(low=50, high=250, name='ffnn_size')
  dim_ffnn_size = Integer(low=30, high=49, name='ffnn_size')
  #dim_contextualization_size = Integer(low=50, high=300, name='contextualization_size')
  dim_contextualization_size = Integer(low=301, high=500, name='contextualization_size')
  dim_decay_rate = Real(low=0.5, high=1,name='decay_rate')

  dimensions = [dim_learning_rate, dim_ffnn_size, dim_contextualization_size, dim_decay_rate]

  @use_named_args(dimensions=dimensions)
  def fitness(learning_rate, ffnn_size, contextualization_size, decay_rate):
      tf.reset_default_graph()
      tf.keras.backend.clear_session()
      name = sys.argv[1] + '_' + str(learning_rate) + '_' + str(ffnn_size) + '_' + str(contextualization_size) + '_' + str(decay_rate)
      config = util.initialize_from_env(True, name, [learning_rate, ffnn_size, contextualization_size, decay_rate])
      print(pyhocon.HOCONConverter.convert(config, "hocon"))
      report_frequency = config["report_frequency"]
      eval_frequency = config["eval_frequency"]
      max_step = config["max_step"]
      train_size = config["dataset_size"]*0.9
      test_size = config["dataset_size"]*0.1
      print('will do model')
      model = biaffine_ner_model.BiaffineNERModel(config)
      print('has  model')
      saver = tf.train.Saver(max_to_keep=1)
          
      print('helloooo')

      log_dir = config["log_dir"]
      writer = tf.summary.FileWriter(log_dir, flush_secs=20)

      max_f1 = 0
      best_step = 0

      session_config = tf.ConfigProto()
      session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
      session_config.gpu_options.allow_growth = True
      session_config.allow_soft_placement = True

      
      with tf.Session(config=session_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        session.run(tf.global_variables_initializer())
        model.start_enqueue_thread(session)
        accumulated_loss = 0.0
        accumulated_len = 0

        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
          print("Restoring from: {}".format(ckpt.model_checkpoint_path))
          saver.restore(session, ckpt.model_checkpoint_path)

        initial_time = time.time()
        while True:
          print('running')

          tf_loss, predictions, tf_global_step, _ = session.run([model.loss, model.predictions, model.global_step, model.train_op])
          accumulated_loss += tf_loss
          accumulated_len += len(predictions)
          print(tf_global_step)
          

          if tf_global_step % report_frequency == 0:
            total_time = time.time() - initial_time
            steps_per_second = tf_global_step / total_time
            print("[{}] tokens loss={:.2f}, steps/s={:.2f}".format(tf_global_step, accumulated_loss/accumulated_len, steps_per_second))
            writer.add_summary(util.make_summary({"loss": accumulated_loss/accumulated_len}), tf_global_step)
            accumulated_loss = 0.0
            accumulated_loss = 0

          if tf_global_step % eval_frequency == 0:
            saver.save(session, os.path.join(log_dir, "model.ckpt"), global_step=tf_global_step)
            if config['eval_path']:
              print('Evaluateeeeeeee')
              eval_summary, eval_f1, eval_precision, eval_recall, _, _, _, _, _, loss, test_predictions = model.evaluate(session, evaluate_loss=True, eval_train_path=False) #doing for eval test
              if eval_f1 > max_f1:
                max_f1 = eval_f1
                best_step = tf_global_step
                util.copy_checkpoint(os.path.join(log_dir, "model.ckpt-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

              writer.add_summary(eval_summary, tf_global_step)
              writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

              print("[{}] evaL_f1={:.2f}, max_f1={:.2f} at step {}".format(tf_global_step, eval_f1 * 100, max_f1 * 100, best_step))
              print("[{}] token test eval_loss={}, steps/s={:.2f}".format(tf_global_step, loss/test_predictions, steps_per_second))
              
              save_csv_eval([eval_f1, eval_precision, eval_recall], tf_global_step,log_dir + 'evalTest')
              save_csv(loss/test_predictions,tf_global_step,log_dir + '_TestTokenLoss')
            else:
                util.copy_checkpoint(os.path.join(log_dir, "model.ckpt-{}".format(tf_global_step)),
                               os.path.join(log_dir, "model.max.ckpt"))


          if max_step > 0 and tf_global_step >= max_step:
            break
       
        coord.request_stop()
        coord.join(threads)
      del model
      tf.reset_default_graph()
      tf.keras.backend.clear_session()
      
      return - max_f1
    
  search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=10)

  print('optimal parameters found using scikit optimize\n')
  print(search_result.x)
  print(search_result.x_iters)
  print(search_result.func_vals)
  plot_convergence(search_result)
  plt.savefig("Converge_" + sys.argv[1] + ".png", dpi=400)     
