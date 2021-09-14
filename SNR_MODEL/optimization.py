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
       # for el in range(0,4):
        #    if len(entities) >= el + 1:
         #       result.append(entities[el])
          #  else:
           #     result.append('NONE')
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
       # for el in range(0,4):
        #    if len(entities) >= el + 1:
         #       result.append(entities[el])
          #  else:
           #     result.append('NONE')
        data_frame.loc[epoch] = entities
        data_frame.to_csv("./" + name + ".csv")

    return 0

#print(datetime.datetime.now())

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"


  dim_learning_rate = Real(low=1e-3, high=1e-1, prior='log-uniform',name='learning_rate')
  dim_ffnn_size = Integer(low=50, high=250, name='ffnn_size')
  dim_contextualization_size = Integer(low=50, high=300, name='contextualization_size')
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
      #session_config.log_device_placement = True
      
      print('helloooo')


      with tf.Session(config=session_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        session.run(tf.global_variables_initializer())
        model.start_enqueue_thread(session)
        accumulated_loss = 0.0
        accumulated_len = 0
        print('helloooo')


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
   
  #NORMS
  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=10,x0=[[0.001, 150, 200, 0.999], [1.676785811330371e-05, 134, 258, 0.5612805695976921], [3.3504223221253144e-05, 167, 181, 0.8615257591657655], [0.003055141482694565, 182, 279, 0.5830123797158854], [0.00023090626075687954, 202, 113, 0.967046036094084], [1.2126574543010434e-06, 141, 266, 0.7045836938661888], [6.390013399612975e-06, 179, 296, 0.8545225700591443], [3.477991615712679e-05, 192, 214, 0.7928050560942232], [1.3332711585583025e-06, 196, 167, 0.7279195157125122], [1.2718882899831374e-06, 246, 113, 0.5553788371414152], [0.09163073758725221, 108, 200, 0.6560895711743421], [0.001741257460439244, 163, 286, 0.7556897290901289], [8.754919484776069e-05, 228, 153, 1.0], [0.006909305058261026, 164, 73, 1.0], [0.0117252794157685, 166, 75, 0.5], [0.0003701199680872479, 250, 50, 0.5], [0.0027926312661228025, 250, 288, 1.0], [0.004184820851929094, 50, 50, 0.8073512864051535], [0.026220579996778282, 77, 300, 1.0], [0.0004915811164504759, 250, 300, 1.0]], y0 = [-0.8271237, 0.0,-0.59862779, -0.80929487, -0.80613027, 0.0, 0.0, -0.57754011, 0.0, 0.0, -0.58709107, -0.80347277, -0.7952, -0.80906149, -0.68993506, -0.62592898, -0.83294842, -0.80922099, -0.62959935, -0.81775701])
  
  #NE
  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=20,x0=[[0.001,150,200, 0.999],[0.001305394723134163, 176, 159, 0.627882821435396],[3.0424722373023154e-06, 133, 88, 0.6344839290665234], [0.01615366334979527,156,156,0.9270563063649822],[4.888984594835698e-06,142,179,0.9266646433060634],[0.0009255344580986736,133,167,0.8030305406633181], [0.00029155390292593796,177,116,0.9134354050135043], [0.00022526272329325466,133,189,0.9398399578533907], [0.0005229689376338718,160,81,0.8569220794618221], [0.0006090230349126278,219,172,0.8223281535355249]], y0 = [-0.845119812974868, -0.823394495412844, 0.0, -0.8208430913348947, 0.0, -0.8419825072886298, -0.8150289017341039, -0.8262370540851554, -0.8124645892351274, -0.8319907940161105])
  #SR
 
  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=10,x0=[[0.001, 150, 200, 0.999], [0.000357532354185295, 224, 208, 0.9512993112058217], [0.025654391877211847, 234, 289, 0.9013127149958524], [5.670638673386042e-05, 109, 64, 0.9438811321747607], [8.636290568212024e-06, 109, 159, 0.8067242386430342], [0.057250111758077996, 51, 104, 0.9373958591648617], [2.3702940477655685e-05, 87, 157, 0.955727731564386], [0.0035673582269284965, 119, 107, 0.7659426246198158], [3.56291103691109e-05, 187, 74, 0.9104987216763936], [0.018680229043785593, 247, 246, 0.6078482022903883], [0.008800010715611319, 115, 196, 0.8063136502553654], [0.0013574768286813085, 50, 300, 0.5], [0.007595943462014171, 50, 300, 1.0], [0.0009252158218224993, 250, 293, 1.0], [0.00036571240148029554, 50, 97, 1.0], [0.002784201909285334, 250, 50, 1.0], [0.0007820555963189352, 50, 50, 1.0], [0.007674089651954778, 149, 50, 1.0], [0.0005895011131148597, 182, 50, 1.0], [0.0007415470223965072, 250, 50, 0.5]], y0 = [-0.73407056, -0.72497366, -0.64041096, -0.31684761,  0.0, -0.59236453, -0.10970464, -0.71501204, -0.04500978, -0.59645853, -0.74034335, -0.66666667, -0.72017962, -0.73393045, -0.70323264, -0.72051282, -0.71046229, -0.71603053, -0.70854678, -0.55522972])
  #NE with norms
  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=29,x0=[[0.001,150,200, 0.999]], y0 = [-0.851917404129793])
  #SR with norms
  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=29,x0=[[0.001,150,200, 0.999]], y0 = [-0.759867583396995])
  
  
  #NORMS
  #pt_law_10_NORMS_26_07_opt_32_0.02423900653612488_138_205_0.8034692393895433 - 1
  #pt_law_10_NORMS_26_07_opt_32_0.07267652752883945_219_237_0.8931722606410599 - 2
  #pt_law_10_NORMS_26_07_opt_32_0.002633499164162156_104_56_0.8768565277497855 - 3
  #pt_law_10_NORMS_26_07_opt_32_0.001448307275332236_104_195_0.501810916397751 - 4
  #pt_law_10_NORMS_26_07_opt_32_0.0028629611602629377_93_121_0.9229602381157702 - 5
  #pt_law_10_NORMS_26_07_opt_32_0.009225743999705759_175_79_0.5351703335431729 - 6
  #pt_law_10_NORMS_26_07_opt_32_0.005553099298213195_165_228_0.6293495514683931 - 7
  #pt_law_10_NORMS_26_07_opt_32_0.04429400494132821_115_139_0.6421194226770716 - 8
  #pt_law_10_NORMS_26_07_opt_32_0.023159522200063096_164_80_0.9482397510400009 - 9
  #pt_law_10_NORMS_26_07_opt_32_0.01909860892357006_129_152_0.5807164522356729 - 10
  #pt_law_10_NORMS_26_07_opt_32_0.00924937723910393_233_50_1.0 - 11
  


  
  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=18,x0=[[0.001,150,200, 0.999],[0.02423900653612488,138,205,0.8034692393895433], [0.07267652752883945,219,237,0.8931722606410599], [0.002633499164162156,104,56,0.8768565277497855],[0.001448307275332236,104,195,0.501810916397751], [0.0028629611602629377,93,121,0.9229602381157702], [0.009225743999705759,175,79,0.5351703335431729],  [0.005553099298213195,165,228,0.6293495514683931], [0.04429400494132821,115,139,0.6421194226770716], [0.023159522200063096,164,80,0.9482397510400009], [0.01909860892357006,129,152,0.5807164522356729],  [0.00924937723910393,233,50,1.0]], y0 = [-0.8271237, -0.6808163265306123, -0.6263286999182338, -0.8166797797010228, -0.7906976744186047, -0.8295719844357977, -0.7722132471728596, -0.8075412411626081, -0.6460032626427405, -0.8062142273098937, -0.6965742251223491, -0.8082083662194159])
  
  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=20,x0=[[0.001, 150, 200, 0.999], [0.02423900653612488, 138, 205, 0.8034692393895433], [0.07267652752883945, 219, 237, 0.8931722606410599], [0.002633499164162156, 104, 56, 0.8768565277497855], [0.001448307275332236, 104, 195, 0.501810916397751], [0.0028629611602629377, 93, 121, 0.9229602381157702], [0.009225743999705759, 175, 79, 0.5351703335431729], [0.005553099298213195, 165, 228, 0.6293495514683931], [0.04429400494132821, 115, 139, 0.6421194226770716], [0.023159522200063096, 164, 80, 0.9482397510400009], [0.01909860892357006, 129, 152, 0.5807164522356729], [0.00924937723910393, 233, 50, 1.0], [0.005474930290028525, 181, 104, 0.7248737157568105], [0.09541115013324755, 59, 66, 0.8297054375906233], [0.0702871790231733, 179, 185, 0.9323146256298612], [0.007943010006907081, 144, 196, 0.9525988169642057], [0.001186428759827467, 107, 300, 0.6491899049505832], [0.004215144328560204, 195, 131, 0.7984912255878125], [0.0010991431393342784, 169, 141, 0.5942209242718379], [0.011641257866464392, 76, 284, 0.8355066605460645], [0.002080004215086348, 171, 178, 0.6821377716948549], [0.06644971530000038, 232, 170, 0.6727846877387273], [0.006778571678401113, 50, 300, 0.8656666356174549], [0.001, 50, 300, 0.8560619735876909], [0.015112205154280545, 50, 300, 1.0], [0.009199271385496357, 204, 300, 1.0], [0.008809999695794416, 250, 50, 0.878219310318779], [0.0019942057180530634, 183, 253, 1.0], [0.001, 243, 65, 0.9098055701068262], [0.005311537922315577, 50, 300, 0.864182470495047]], y0 = [-0.8271237,  -0.68081633, -0.6263287,  -0.81667978, -0.79069767, -0.82957198, -0.77221325, -0.80754124, -0.64600326, -0.80621423, -0.69657423, -0.80820837, -0.80895284, -0.56284153, -0.62908497, -0.82753287, -0.80414013, -0.83102919, -0.78406375, -0.82867925, -0.81175536, -0.62857143, -0.83921569, -0.83218391, -0.79149632, -0.80390561, -0.8153967,  -0.81638847, -0.81068342, -0.83830455])
  
  
  #NE
  
  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=29,x0=[[0.001,150,200, 0.999]], y0 = [-0.845119812974868])
  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=20,x0=[[0.001, 150, 200, 0.999], [0.002112038582321693, 152, 54, 0.9338416473582855], [0.0034717670336597306, 214, 80, 0.814365976847168], [0.0021347819380605135, 228, 178, 0.6147652393131846], [0.0042441413312466125, 75, 252, 0.8581906831865003], [0.0020215780679044937, 149, 299, 0.9880080408863902], [0.010183937779554956, 60, 149, 0.954154748368373], [0.026308175165268708, 225, 263, 0.7574808733038481], [0.0015576512702164696, 167, 277, 0.8524278813875635], [0.07859344025476182, 79, 212, 0.6686633847326235], [0.03544029847044905, 90, 280, 0.5486092614318355], [0.001, 50, 300, 0.9419279184162905], [0.06529828104170128, 250, 300, 1.0], [0.001, 50, 300, 0.5], [0.018455877872014366, 250, 254, 1.0], [0.001, 50, 300, 0.6831838674383617], [0.010473608831536729, 50, 300, 0.8308863800864856], [0.00595727390138861, 219, 287, 0.5], [0.001245495357704435, 104, 289, 1.0], [0.0029611893511501085, 50, 300, 0.5], [0.006148594748070925, 50, 300, 0.6506153709943996], [0.005997669294242812, 250, 300, 0.9064475312431364], [0.012256337165453901, 103, 186, 0.5], [0.007737438456869853, 50, 300, 1.0], [0.004840196312983251, 50, 50, 1.0], [0.002546702135894514, 71, 279, 1.0], [0.003951947189611815, 63, 300, 1.0], [0.009210389894972196, 250, 300, 0.5], [0.006800478370967059, 250, 50, 1.0], [0.004173449094844068, 250, 50, 0.5]], y0 = [-0.84511981, -0.84820921, -0.84897959, -0.83419392, -0.8675535,  -0.84216524, -0.84465446, -0.78084715, -0.85909869, -0.35314685, -0.30261882, -0.86042504, -0.37937872, -0.81102814, -0.72682927, -0.84014002, -0.86398132, -0.82842288, -0.84597433, -0.83969466, -0.84420929, -0.84943011, -0.82183563, -0.84270953, -0.83451537, -0.85159011, -0.84241359, -0.79089791, -0.83833718, -0.81512109])

  #SR
  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=20,x0=[[0.001, 150, 200, 0.999], [0.016628108661634593, 144, 139, 0.5489350966562777], [0.0021609988892547245, 93, 54, 0.9438800121344812], [0.017322951589416283, 244, 82, 0.9974801524857386], [0.005706502180437914, 190, 99, 0.674147818975756], [0.004672016219943546, 135, 63, 0.796592616872885], [0.028759725370779615, 245, 165, 0.560949751003618], [0.005777645381858156, 163, 92, 0.8512671168352388], [0.005332968728548248, 225, 99, 0.9279587644445474], [0.01096007270664905, 164, 146, 0.7958542830731916], [0.0068897545485747365, 102, 173, 0.5542378031703983], [0.007897744601006297, 250, 50, 1.0], [0.0059449823524667265, 164, 287, 1.0], [0.001, 235, 86, 1.0], [0.04279693561536906, 50, 60, 1.0], [0.001, 73, 300, 0.6113695929121833], [0.0014917800955853958, 174, 230, 1.0], [0.0034903067856396964, 81, 300, 1.0], [0.001, 50, 93, 1.0], [0.007104142864242049, 195, 300, 0.837087429221989], [0.1, 117, 300, 0.5], [0.0021083914550706897, 250, 300, 0.5], [0.001925576961622797, 250, 300, 0.8104983132673114], [0.0022061290655594743, 250, 300, 1.0], [0.021813547678492903, 50, 300, 0.8019089908383021], [0.004812618402801099, 250, 235, 0.8387730857155411], [0.0014515719262783173, 250, 156, 0.8038380673809724], [0.035787731265611064, 50, 50, 0.5], [0.004217119075734874, 250, 300, 0.811930725541437], [0.0070517040418782975, 250, 202, 0.8718646196282249]], y0 = [-0.73407056, -0.68797737, -0.7321334,  -0.68806584, -0.71928879, -0.71972318, -0.62957453, -0.73703226, -0.73714891, -0.73374038, -0.70448912, -0.70878113, -0.74361647, -0.72901921, -0.66355909, -0.68588362, -0.73219448, -0.73533246, -0.72616847, -0.74445618,  0.0,         -0.70129171, -0.7445294,  -0.72779221, -0.57773512, -0.74473068, -0.73493024, -0.62958509, -0.74074074, -0.74502521])
  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=29,x0=[[0.001,150,200, 0.999]], y0 = [-0.73407056])
  #NE WITH NORMS
  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=29,x0=[[0.001,150,200, 0.999]], y0 = [-0.851917404129793])
  search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=20,x0=[[0.001, 150, 200, 0.999], [0.028897727660629573, 245, 59, 0.7579279774598522], [0.042326844296646626, 228, 129, 0.8849754533803929], [0.07342096880421152, 162, 146, 0.5656083766684391], [0.0018493663625436898, 82, 282, 0.8855826664409727], [0.004531165342543305, 54, 284, 0.8718741293739042], [0.017354037308295385, 101, 88, 0.5882372105003779], [0.009879689419542483, 74, 277, 0.5837959877120615], [0.0010145902067357354, 181, 250, 0.5799438155595796], [0.012693595652811085, 97, 146, 0.5784875348377975], [0.011688034221682722, 221, 195, 0.970218434804526], [0.0028668452420429974, 50, 50, 1.0], [0.001, 242, 256, 0.9519076870111983], [0.0061728834434303164, 50, 50, 0.5], [0.0013758627569471237, 50, 50, 1.0], [0.003856472182009422, 50, 300, 0.5], [0.0021193006183416467, 50, 50, 1.0], [0.0036673156319169016, 50, 300, 1.0], [0.02923316928534802, 50, 300, 1.0], [0.005429081379500235, 250, 275, 1.0], [0.001, 118, 186, 1.0], [0.003518199593516861, 156, 300, 1.0], [0.010557653019306517, 50, 50, 0.5], [0.001954921500809089, 250, 300, 1.0], [0.014192355084474721, 250, 300, 0.80554436136662], [0.0013647902148945673, 50, 50, 0.5], [0.002346325909340456, 250, 300, 0.5963991567206675], [0.027260626388321385, 250, 300, 1.0], [0.03384646340317416, 50, 50, 0.5399283146417562], [0.08096624729733061, 50, 280, 0.5003466719886219]], y0 = [-0.8519174,  -0.84521739, -0.75064599,  0.0,         -0.86083382, -0.86171429, -0.83920188, -0.83548766, -0.81667619, -0.83976608, -0.81468732, -0.84518828, -0.84619849, -0.79976233, -0.84502262, -0.82862297, -0.83704149, -0.84852002, -0.52211127, -0.83352736, -0.84773663, -0.85259434, -0.82276995, -0.84994401, -0.82242991, -0.72875,    -0.83680556, -0.34066986, -0.81253696, -0.31309298])
 
  #SR WITH NORMS
  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=20,x0= [[0.001, 150, 200, 0.999], [0.002550488705013783, 191, 182, 0.6486524272052533], [0.0467214364350893, 95, 267, 0.6159306098046401], [0.0048886931885861774, 90, 289, 0.5103451764597475], [0.00508125636281769, 134, 202, 0.5834921674828942], [0.012491765030010444, 141, 163, 0.9121565234400455], [0.029948936665740406, 224, 114, 0.5145641500439269], [0.02260660228486574, 211, 292, 0.8743601256108473], [0.021202745143058536, 220, 88, 0.9597325828319551], [0.0012708354727404057, 236, 96, 0.9556117849264412], [0.0018106857228353933, 227, 148, 0.6462117490477607], [0.011832650617002317, 250, 50, 0.5576202106795308], [0.001971295966933988, 50, 50, 1.0], [0.007699891421322025, 103, 50, 1.0], [0.001, 50, 50, 0.5], [0.002321727863944083, 117, 300, 1.0], [0.0032767397771356495, 174, 64, 0.9241435543827969], [0.006149757601914804, 195, 217, 0.9398691066063583], [0.010523818856267013, 174, 50, 1.0], [0.0036343635396982087, 50, 50, 1.0], [0.001, 115, 281, 0.9855143335618357], [0.015922352868666483, 50, 50, 1.0], [0.010903785833911835, 50, 300, 1.0], [0.001, 250, 300, 1.0], [0.001, 50, 300, 0.8018422097004836], [0.1, 50, 300, 1.0], [0.0014029718116045818, 248, 215, 1.0], [0.019556980427433843, 250, 50, 0.6930639802376473], [0.0037468182219759795, 250, 300, 1.0], [0.013830790625960447, 213, 300, 0.5]], y0 = [-0.75986758, -0.73499734, -0.00100301, -0.73199783, -0.73633441, -0.75940825, -0.60866948, -0.64769323, -0.73375474, -0.75221463, -0.72152907, -0.65985816, -0.74800515, -0.75104603, -0.49285228, -0.7535545,  -0.75854813, -0.76287554, -0.71604938, -0.73192191, -0.75435005, -0.72694945, -0.72653503, -0.74926254, -0.74441153,  0.0,         -0.75514019, -0.71145804, -0.7467584,  -0.65521258])

  #search_result = gp_minimize(func=fitness,dimensions=dimensions,acq_func='EI',n_calls=29,x0=[[0.001,150,200, 0.999]], y0 = [-0.759867583396995])

  print('optimal parameters found using scikit optimize\n')
  print(search_result.x)
  print(search_result.x_iters)
  print(search_result.func_vals)
  plot_convergence(search_result)
  plt.savefig("Converge_" + sys.argv[1] + ".png", dpi=400)
            
  '''
file = open("pt_law_10_NORMS_26_07_opt_32_optpy.txt", "r")
l = file.readlines()
file.close()
exp = 0
for line in l:
    if "Running experiment: pt_law_10_NORMS_26_07_opt_32" in line:
        print(line)
        exp += 1
        print(exp)
        
file = open("pt_law_10_NE_withNorms_26_07_opt_32_optpy.txt", "r")
l = file.readlines()
file.close()
exp = 0
for line in l:
    if "Running experiment: pt_law_10_NE_withNorms_26_07_opt_32" in line:
        print(line)
        exp += 1
        print(exp)
        
xDZ8mpnj
        
   
6nd: pt_law_10_SR_withNorms_26_07_opt_32_default.txt done need to add to excel


optimizar com 16 (nao vai passar 15) e com learning rate modificada NORMS -- 30 combinações done
optimizar com 15 (nao vai passar 15) e com learning rate modificada NE -- 30 combinações doing


#optimizar com 20 e com learning rate modificada -- 30 combinações

#o ne with norms devia ser 20 ou 15? deveria fazer combinações para verificar?
# e o sr sem norms? devia ser 15? ou 20? devia fazer combinações para verificar ???
#preciso da maquina livre senao não consigo correr tudo.... demora imenso



        


  '''
    
