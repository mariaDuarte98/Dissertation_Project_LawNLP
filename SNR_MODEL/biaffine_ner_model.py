from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,time,json,threading
import random, io
import numpy as np
#import larq as lq
from collections import defaultdict
#import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()
import h5py
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import util


class BiaffineNERModel():
  def __init__(self, config):
    self.config = config
    #self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
    #self.context_embeddings_size = self.context_embeddings.size

    #self.char_embedding_size = config["char_embedding_size"]
    #self.char_dict = util.load_char_dict(config["char_vocab_path"])

    print('create model')
    print(self.config["lm_path"])
    if self.config["lm_path"].lower() == "none":
      self.lm_file = None
    else:
      self.lm_file = h5py.File(self.config["lm_path"], "r")
    self.lm_layers = self.config["lm_layers"]
    self.lm_size = self.config["lm_size"]

    self.eval_data = None  # Load eval data lazily.
    self.ner_types = self.config['ner_types']
    self.ner_maps = {ner: (i + 1) for i, ner in enumerate(self.ner_types)}
    self.num_types = len(self.ner_types)

    input_props = []
    input_props.append((tf.string, [None, None]))  # Tokens.
  #  input_props.append((tf.float32, [None, None, self.context_embeddings_size]))  # Context embeddings.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers]))  # LM embeddings.
  #  input_props.append((tf.int32, [None, None, None]))  # Character indices.
    input_props.append((tf.int32, [None]))  # Text lengths.
    input_props.append((tf.bool, []))  # Is training.
    input_props.append((tf.int32, [None]))  # Gold NER Label

    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()

    self.predictions, self.loss = self.get_predictions_and_loss(self.input_tensors)
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)
    learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                               self.config["decay_frequency"], self.config["decay_rate"],
                                               staircase=True)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    optimizers = {
      "adam": tf.train.AdamOptimizer,
      "sgd": tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[self.config["optimizer"]](learning_rate)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)
    
    print('done model')

  def start_enqueue_thread(self, session):
    gold_labels_count = 0
    with open(self.config["train_path"]) as f:
      train_examples = [json.loads(jsonline) for jsonline in f.readlines()]


      
    def _enqueue_loop():
      batch_size = self.config["batch_size"]
      while True:
        random.shuffle(train_examples) #using static batches, would it not be better to use dynamic batches?
        batches = [train_examples[x:x+batch_size] for x in range(0, len(train_examples), batch_size)]
        # ex1 ex2 .... exn -> suffle and divide in batches
        for batch in batches:
          tensorized_example = self.tensorize_example(batch, is_training=True)
          feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
          session.run(self.enqueue_op, feed_dict=feed_dict)
          
    #q = Queue.Queue()
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()
    '''for example in train_examples:
      gold_labels = []
      for sid, sent in enumerate(example["sentences"]):
        ner = {(s,e):self.ner_maps[t] for s,e,t in example["ners"][sid]}
        for s in range(len(sent)):
          for e in range(s,len(sent)):
            gold_labels.append(ner.get((s,e),0))
      gold_labels_count += len(gold_labels)
    #enqueue_thread.result_queue = q
    return gold_labels_count'''
    

  def restore(self, session):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
    saver = tf.train.Saver(vars_to_restore)
    checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)

  def load_lm_embeddings(self, ids):
    if self.lm_file is None:
      return np.zeros([0, 0, self.lm_size, self.lm_layers])

    sentences = []
    for id in ids:
        sentences.append(self.lm_file[id])
    num_sentences = len(ids)
    lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
    for i, s in enumerate(sentences):
      lm_emb[i, :s.shape[0], :, :] = s
    return lm_emb

  def tensorize_example(self, example, is_training, evaluate_loss=False):
    ners = []
    sentences = []
    ids = []
    for sent in example:
        ners.append(sent["ners"])
        sentences.append(sent["sentence"])
        ids.append(sent["id"])
    
    
    #ners = example["ners"]
    #sentences = example["sentences"]

    
    max_sentence_length = max(len(s) for s in sentences)
    #max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
    text_len = np.array([len(s) for s in sentences])
    tokens = [[""] * max_sentence_length for _ in sentences]
    


    tokens = np.array(tokens)

    #doc_key = example["doc_key"]

    lm_emb = self.load_lm_embeddings(ids)

    gold_labels = []
    if is_training or evaluate_loss:
      for sid, sent in enumerate(sentences):
        ner = {(s,e):self.ner_maps[t] for s,e,t in ners[sid]}
        for s in range(len(sent)):
          for e in range(s,len(sent)):
            gold_labels.append(ner.get((s,e),0))
    gold_labels = np.array(gold_labels)

    #example_tensors = (tokens, context_word_emb,lm_emb, char_index, text_len, is_training, gold_labels)
    if evaluate_loss:
        example_tensors = (tokens, lm_emb, text_len, False, gold_labels)
    else:
        example_tensors = (tokens, lm_emb, text_len, is_training, gold_labels)
    
    return example_tensors

  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)

  def lstm_contextualize(self, text_emb, text_len, lstm_dropout):
    num_sentences = tf.shape(text_emb)[0]

    current_inputs = text_emb  # [num_sentences, max_sentence_length, emb]

    for layer in range(self.config["contextualization_layers"]):
      with tf.variable_scope("layer_{}".format(layer), reuse=tf.AUTO_REUSE):
        with tf.variable_scope("fw_cell"):
          cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, lstm_dropout)
        with tf.variable_scope("bw_cell"):
          cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, lstm_dropout)
        state_fw = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
          tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
          tf.tile(cell_fw.initial_state.h, [num_sentences, 1]),
        )
        state_bw = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
          tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
          tf.tile(cell_bw.initial_state.h, [num_sentences, 1]),
        )

        (fw_outputs, bw_outputs), ((_, fw_final_state), (_, bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=current_inputs,
          sequence_length=text_len,
          initial_state_fw=state_fw,
          initial_state_bw=state_bw)

        text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [num_sentences, max_sentence_length, emb]
        text_outputs = tf.nn.dropout(text_outputs, lstm_dropout)
        if layer > 0:
          highway_gates = tf.sigmoid(
            util.projection(text_outputs, util.shape(text_outputs, 2)))  # [num_sentences, max_sentence_length, emb]
          text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
        current_inputs = text_outputs

    return text_outputs

  def get_predictions_and_loss(self, inputs):
    #tokens, context_word_emb, lm_emb, char_index, text_len, is_training, gold_labels = inputs
    tokens, lm_emb, text_len, is_training, gold_labels = inputs
    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
    self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
    self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

    num_sentences = tf.shape(tokens)[0]
    max_sentence_length = tf.shape(tokens)[1]

    context_emb_list = []
    #context_emb_list.append(context_word_emb)
    #char_emb = tf.gather(tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), char_index) # [num_sentences, max_sentence_length, max_word_length, emb]
    #flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), util.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
    #flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
    #aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
    #context_emb_list.append(aggregated_char_emb)


    if self.lm_file is not None:  # Only add these layers if we're using contextualized embeddings
      lm_emb_size = util.shape(lm_emb, 2)
      lm_num_layers = util.shape(lm_emb, 3)
      with tf.variable_scope("lm_aggregation"):
        self.lm_weights = tf.nn.softmax(tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
        self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))

      flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
      flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights, 1)) # [num_sentences * max_sentence_length * emb, 1]
      aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
      aggregated_lm_emb *= self.lm_scaling
      context_emb_list.append(aggregated_lm_emb)

    context_emb = tf.concat(context_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    context_emb = tf.nn.dropout(context_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]

    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length) # [num_sentence, max_sentence_length]

    candidate_scores_mask = tf.logical_and(tf.expand_dims(text_len_mask,[1]),tf.expand_dims(text_len_mask,[2])) #[num_sentence, max_sentence_length,max_sentence_length]
    sentence_ends_leq_starts = tf.tile(tf.expand_dims(tf.logical_not(tf.sequence_mask(tf.range(max_sentence_length),max_sentence_length)), 0),[num_sentences,1,1]) #[num_sentence, max_sentence_length,max_sentence_length]
    candidate_scores_mask = tf.logical_and(candidate_scores_mask,sentence_ends_leq_starts)

    flattened_candidate_scores_mask = tf.reshape(candidate_scores_mask,[-1]) #[num_sentence * max_sentence_length * max_sentence_length]


    context_outputs = self.lstm_contextualize(context_emb, text_len,self.lstm_dropout) # [num_sentence, max_sentence_length, emb]


    with tf.variable_scope("candidate_starts_ffnn"):
      candidate_starts_emb = util.projection(context_outputs,self.config["ffnn_size"]) #[num_sentences, max_sentences_length,emb]
    with tf.variable_scope("candidate_ends_ffnn"):
      candidate_ends_emb = util.projection(context_outputs,self.config["ffnn_size"]) #[num_sentences, max_sentences_length, emb]


    candidate_ner_scores = util.bilinear_classifier(candidate_starts_emb,candidate_ends_emb,self.dropout,output_size=self.num_types+1)#[num_sentence, max_sentence_length,max_sentence_length,types+1]
    candidate_ner_scores = tf.boolean_mask(tf.reshape(candidate_ner_scores,[-1,self.num_types+1]),flattened_candidate_scores_mask)
   

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gold_labels, logits=candidate_ner_scores)
    loss = tf.reduce_sum(loss)
    
    
    return candidate_ner_scores, loss



  def get_pred_ner(self, sentences, span_scores, is_flat_ner):
    candidates = []
    for sid,sent in enumerate(sentences):
      for s in range(len(sent)):
        for e in range(s,len(sent)):
          candidates.append((sid,s,e))

    top_spans = [[] for _ in range(len(sentences))]
    for i, type in enumerate(np.argmax(span_scores,axis=1)):
      if type > 0:
        sid, s,e = candidates[i]
        top_spans[sid].append((s,e,type,span_scores[i,type]))


    top_spans = [sorted(top_span,reverse=True,key=lambda x:x[3]) for top_span in top_spans]
    sent_pred_mentions = [[] for _ in range(len(sentences))]
    for sid, top_span in enumerate(top_spans):
      for ns,ne,t,_ in top_span:
        for ts,te,_ in sent_pred_mentions[sid]:
          if ns < ts <= ne < te or ts < ns <= te < ne:
            #for both nested and flat ner no clash is allowed
            break
          if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
            #for flat ner nested mentions are not allowed
            break
        else:
          sent_pred_mentions[sid].append((ns,ne,t))
    pred_mentions = set((sid,s,e,t) for sid, spr in enumerate(sent_pred_mentions) for s,e,t in spr)
    return pred_mentions

  def load_eval_data(self, is_training=False, evaluate_loss=False, eval_train_path=False):
    #if self.eval_data is None:
    def load_line(batch):
      return self.tensorize_example(batch, is_training, evaluate_loss), batch
    
    if not eval_train_path:
        eval = self.config["eval_path"]
    else:
        eval = self.config["eval_path_train"]
    with open(eval) as f:
      eval_examples = [json.loads(jsonline) for jsonline in f.readlines()]
      batch_size = self.config["batch_size"]
      batches = [eval_examples[x:x+batch_size] for x in range(0, len(eval_examples), batch_size)]
      self.eval_data = [load_line(batch) for batch in batches]

    print(eval)
    print("Loaded {} eval examples.".format(len(self.eval_data)))

# Agreement between gold labels and predicted labels #
  def agreement(self, gold_ners, pred_ners, sentences_ids, sentences):
    gold_sentences = {}
    pred_sentences = {}
    batch_count = 0
    for batch in gold_ners:
        for ner in batch: # ner = (sentid, start, end, label)
            if sentences_ids[batch_count][ner[0]] not in gold_sentences.keys():
                gold_sentences[sentences_ids[batch_count][ner[0]]] = {}
            for index in range(ner[1],ner[2] + 1):
                if index not in gold_sentences[sentences_ids[batch_count][ner[0]]].keys():
                    gold_sentences[sentences_ids[batch_count][ner[0]]][index] = []
                gold_sentences[sentences_ids[batch_count][ner[0]]][index].append(ner[3])
        batch_count += 1
            
            
    batch_count = 0
    for batch in pred_ners:
        for ner in batch: # ner = (sentid, start, end, label)
            if sentences_ids[batch_count][ner[0]] not in pred_sentences.keys():
                pred_sentences[sentences_ids[batch_count][ner[0]]] = {}
            for index in range(ner[1],ner[2] + 1):
                if index not in pred_sentences[sentences_ids[batch_count][ner[0]]].keys():
                    pred_sentences[sentences_ids[batch_count][ner[0]]][index] = []
                pred_sentences[sentences_ids[batch_count][ner[0]]][index].append(ner[3])
        batch_count += 1
     

    for sentId in gold_sentences.keys():
        for index in gold_sentences[sentId].keys():
            gold_sentences[sentId][index] = sorted( gold_sentences[sentId][index])
    


    for sentId in pred_sentences.keys():
        for index in pred_sentences[sentId].keys():
            pred_sentences[sentId][index] = sorted(pred_sentences[sentId][index])
            
    avg_score = 0
    batch_count = 0
    sentences_num = 0
    for batch in sentences:
        sent_count = 0
        for sentence in batch:
            sentences_num += 1
            sent_score = 0
            num_tokens = len(sentence)
            sent_id = sentences_ids[batch_count][sent_count]
            if sent_id not in  gold_sentences.keys():
                gold_sentences[sent_id] = {}
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
                #elif gold_sentences[sent_id][i] == pred_sentences[sent_id][i]:
                elif len(list(set(gold_sentences[sent_id][i]).intersection(pred_sentences[sent_id][i]))) > 0:
                    sent_score += len(list(set(gold_sentences[sent_id][i]).intersection(pred_sentences[sent_id][i]))) / max(len(gold_sentences[sent_id][i]), len(pred_sentences[sent_id][i]))
                    #sent_score += 1
                else:
                    sent_score += 0
            sent_score = sent_score / num_tokens
            avg_score += sent_score
            sent_count += 1
        batch_count += 1
    avg_score /= sentences_num
    
    return avg_score
    
# Agreement for each label between gold labels and predicted labels #
  def agreement_by_label(self, gold_ners, pred_ners, sentences, dict_labels):
    labels_indexes = dict_labels.values()
    gold_sentences = {}
    pred_sentences = {}
    avg_score = {}
    for label in labels_indexes:
        batch_count = 0
        has_label = {}
        has_label_gold = 0
        gold_sentences[label] = {}
        pred_sentences[label] = {}
        for batch in gold_ners:
            has_label[batch_count] = []
            gold_sentences[label][batch_count] = {}
            for ner in batch: # ner = (sentid, start, end, label)
                if ner[3] == label:
                    if ner[0] not in has_label[batch_count]:
                        has_label[batch_count].append(ner[0])
                    has_label_gold += 1
                    if ner[0] not in gold_sentences[label][batch_count].keys():
                        gold_sentences[label][batch_count][ner[0]] = {}
                    for index in range(int(ner[1]),int(ner[2]) + 1):
                        if index not in gold_sentences[label][batch_count][ner[0]].keys():
                            gold_sentences[label][batch_count][ner[0]][index] = []
                        gold_sentences[label][batch_count][ner[0]][index].append(ner[3])
            batch_count += 1
        batch_count = 0
        for batch in pred_ners:
            pred_sentences[label][batch_count] = {}
            for ner in batch: # ner = (sentid, start, end, label)
                if ner[3] == label:
                    if ner[0] not in has_label[batch_count]:
                        has_label[batch_count].append(ner[0])
                    if ner[0] not in pred_sentences[label][batch_count].keys():
                            pred_sentences[label][batch_count][ner[0]] = {}
                    for index in range(int(ner[1]),int(ner[2]) + 1):
                        if index not in pred_sentences[label][batch_count][ner[0]].keys():
                                pred_sentences[label][batch_count][ner[0]][index] = []
                        pred_sentences[label][batch_count][ner[0]][index].append(ner[3])
            batch_count += 1
                
        avg_score[label] = 0
        #sent_id = 0
        num_has_label = 0
        for batch_count in has_label.keys():
            for sent_id in has_label[batch_count]:
                num_has_label += 1
                sent_score = 0
              #  if label == dict_labels['SCOPE']:
                #    print(sentences[sent_id])
                num_tokens = len(sentences[batch_count][sent_id])
                seen_tokens = 0
                for i in range(0, num_tokens):
                    if sent_id not in gold_sentences[label][batch_count].keys():
                        gold_sentences[label][batch_count][sent_id] = {}
                    if sent_id not in pred_sentences[label][batch_count].keys():
                        pred_sentences[label][batch_count][sent_id] = {}
                    if i not in gold_sentences[label][batch_count][sent_id].keys():
                        gold_sentences[label][batch_count][sent_id][i] = []
                    if i not in pred_sentences[label][batch_count][sent_id].keys():
                        pred_sentences[label][batch_count][sent_id][i] = []
                      
                    gold_sentences[label][batch_count][sent_id][i] = sorted(gold_sentences[label][batch_count][sent_id][i])
                    pred_sentences[label][batch_count][sent_id][i] = sorted(pred_sentences[label][batch_count][sent_id][i])
                    if gold_sentences[label][batch_count][sent_id][i] == [] and pred_sentences[label][batch_count][sent_id][i] == []:
                        sent_score += 0
                        #seen_tokens += 1
                    elif gold_sentences[label][batch_count][sent_id][i] == pred_sentences[label][batch_count][sent_id][i]:
                    #elif len(list(set(gold_sentences[label][batch_count][sent_id][i]).intersection(pred_sentences[label][batch_count][sent_id][i]))) > 0:
                        #sent_score += len(list(set(gold_sentences[label][batch_count][sent_id][i]).intersection(pred_sentences[label][batch_count][sent_id][i]))) / max(len(gold_sentences[label][batch_count][sent_id][i]), len(pred_sentences[label][batch_count][sent_id][i]))
                        sent_score += 1
                        seen_tokens += 1
                    else:
                        sent_score += 0
                        seen_tokens += 1
              #  if label == dict_labels['SCOPE']:
               #     print( gold_sentences[label][sent_id])
               #     print( pred_sentences[label][sent_id])
               #     print(seen_tokens)
                if seen_tokens != 0:
                    sent_score = sent_score / seen_tokens#num_tokens
                else:
                    sent_score = 0
                avg_score[label] += sent_score
                #sent_id += 1
        if num_has_label != 0:
            avg_score[label] /= num_has_label
        if has_label_gold == 0:
            avg_score[label] = str(avg_score[label]) + '->No sent with the label'
       # if label == dict_labels['SCOPE']:
           # print(has_label)
           # print(has_label_gold)
        
    return avg_score
        
        
    

  def evaluate(self, session, is_final_test=False, is_training=False, evaluate_loss=False, eval_train_path=False):
    self.load_eval_data(is_training, evaluate_loss, eval_train_path)

    tp,fn,fp = 0,0,0
    start_time = time.time()
    num_words = 0
    sub_tp,sub_fn,sub_fp = [0] * self.num_types,[0]*self.num_types, [0]*self.num_types
    total_loss = 0.0

    is_flat_ner = 'flat_ner' in self.config and self.config['flat_ner']
    golden_ners = []
    pred_ners_list = []
    predicted_sentences = []
    predicted_sentences_ids = []
    

    conf_pred = {}
    conf_gold = {}


    #batch_size = 10
    #num = 0
    loss = 0
    total_gold_labels = 0
    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      if evaluate_loss:
        candidate_ner_scores, batch_loss = session.run([self.predictions, self.loss], feed_dict=feed_dict)
        loss += batch_loss



        #total_loss += loss
      else:
        candidate_ner_scores = session.run(self.predictions, feed_dict=feed_dict)
      #num += batch_size

      sentences = []
      ids = []
      ners = []
      for s in example:
        num_words += sum(len(tok) for tok in s["sentence"])
        sentences.append(s["sentence"])
        ids.append(s["id"])
        ners.append(s["ners"])
        
      
    
      # sid = sentence id, s = start, e = end, t = tag, self.ner_maps[t] = tag id
      gold_ners = set([(sid,s,e, self.ner_maps[t]) for sid, ner in enumerate(ners) for s,e,t in ner])
      total_gold_labels += len(candidate_ner_scores)
      pred_ners = self.get_pred_ner(sentences, candidate_ner_scores,is_flat_ner)


      
        
        
      pred_ners_list.append(pred_ners)
      golden_ners.append(gold_ners)
      predicted_sentences.append(sentences)
      predicted_sentences_ids.append(ids)
 
      
      ############### add here code to write predict labes into a file (-> transform tag ids into tag labels)

      tp += len(gold_ners & pred_ners)
      fn += len(gold_ners - pred_ners)
      fp += len(pred_ners - gold_ners)
      
      
      tags = ["NoLabel"]
      if is_final_test:
        for i in range(self.num_types):
          tags.append(self.ner_types[i])
          sub_gm = set((sid,s,e) for sid,s,e,t in gold_ners if t ==i+1)
          sub_pm = set((sid,s,e) for sid,s,e,t in pred_ners if t == i+1)
          sub_tp[i] += len(sub_gm & sub_pm)
          sub_fn[i] += len(sub_gm - sub_pm)
          sub_fp[i] += len(sub_pm - sub_gm)

      for sid,s,e,t in gold_ners:
        s_id = ids[sid]
        if s_id not in conf_gold.keys():
            conf_gold[s_id] = {}
        if str(s) + '+' + str(e) not in conf_gold[s_id].keys():
            conf_gold[s_id][str(s) + '+' + str(e)] = []
        conf_gold[s_id][str(s) + '+' + str(e)].append(self.ner_types[t - 1])
      for sid,s,e,t in pred_ners:
        s_id = ids[sid]
        if s_id not in conf_pred.keys():
            conf_pred[s_id] = {}
        if str(s) + '+' + str(e) not in conf_pred[s_id].keys():
            conf_pred[s_id][str(s) + '+' + str(e)] = []
        conf_pred[s_id][str(s) + '+' + str(e)].append(self.ner_types[t - 1])
        
    
      if example_num % 10 == 0:
        print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

    used_time = time.time() - start_time
    print("Time used: %d second, %.2f w/s " % (used_time, num_words*1.0/used_time))

    if is_final_test and self.config["test_output"]:
        if os.path.exists(self.config["test_output"]):
            os.remove(self.config["test_output"])
        file = open(self.config["test_output"],"w")
        file.write(str(tp) + "," + str(fp) + "," + str(fn)) #TP,FP,FN saving results of the model for the system later calculate the final results
        file.write("\n")
        #file.write(str(golden_ners))
        #file.write("\n")
        #file.write(str(pred_ners_list))
        #file.write("\n")
        #file.write(str(predicted_sentences_ids))
        #file.write("\n")
        #file.write(str(predicted_sentences))
        file.close()
    m_r = 0 if tp == 0 else float(tp)/(tp+fn)
    m_p = 0 if tp == 0 else float(tp)/(tp+fp)
    m_f1 = 0 if m_p == 0 else 2.0*m_r*m_p/(m_r+m_p)
    
    for id in conf_gold.keys():
        if id not in conf_pred.keys():
            conf_pred[id] = {}
        for span in conf_gold[id].keys():
            if span not in conf_pred[id].keys():
                conf_pred[id][span] = ["NoLabel"]
    
    for id in conf_pred.keys():
        if id not in conf_gold.keys():
            conf_gold[id] = {}
        for span in conf_pred[id].keys():
            if span not in conf_gold[id].keys():
                conf_gold[id][span] = ["NoLabel"]
    
    if is_final_test:
        conf_y_gold = []
        conf_y_pred = []
        for id in conf_gold.keys():
            for span in conf_gold[id].keys():
                if len(conf_gold[id][span]) > 1: #cheking segments for erros inn anotation (multi label not allowed)
                    print('opssss')
                    print(conf_gold[id][span])
                    conf_y_gold.append(conf_gold[id][span][1])
                    print(id)
                elif len(conf_gold[id][span]) == 1:
                    conf_y_gold.append(conf_gold[id][span][0])
                if len(conf_pred[id][span]) > 1:  #cheking segments for erros inn anotation (multi label not allowed)
                    print('opsssspred')
                    conf_y_gold.append(conf_gold[id][span][1])
                    print(id)
                elif len(conf_pred[id][span]) == 1:
                    conf_y_pred.append(conf_pred[id][span][0])

            

    
        print(len(conf_y_gold))
        print(len(conf_y_pred))
        print(tags)
        conf_matrix = confusion_matrix(conf_y_gold, conf_y_pred, labels=tags)
        
        print(conf_matrix)
        print(total_gold_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=tags)
        disp.plot()
    agreement_result = self.agreement(golden_ners, pred_ners_list, predicted_sentences_ids, predicted_sentences)
    print("Agreement between gold and predicted NERS: " + str(agreement_result))
    print("Mention F1: {:.2f}%".format(m_f1*100))
    print("Mention recall: {:.2f}%".format(m_r*100))
    print("Mention precision: {:.2f}%".format(m_p*100))
    
    
    
    no_sent_label = []
    # print(no_show)
    if is_final_test:
      result = self.agreement_by_label(golden_ners, pred_ners_list, predicted_sentences, self.ner_maps)
      for label_index in result.keys():
        label = str(list(self.ner_maps.keys())[list(self.ner_maps.values()).index(label_index)])
        print("Agreement of " + label + ": " + str(result[label_index]))
        if 'No sent with the label' in str(result[label_index]):
            no_sent_label.append(label)
      print("****************SUB NER TYPES********************")
      for i in range(self.num_types):
        sub_r = 0 if sub_tp[i] == 0 else float(sub_tp[i]) / (sub_tp[i] + sub_fn[i])
        sub_p = 0 if sub_tp[i] == 0 else float(sub_tp[i]) / (sub_tp[i] + sub_fp[i])
        sub_f1 = 0 if sub_p == 0 else 2.0 * sub_r * sub_p / (sub_r + sub_p)

        print("{} F1: {:.2f}%".format(self.ner_types[i],sub_f1 * 100))
        print("{} recall: {:.2f}%".format(self.ner_types[i],sub_r * 100))
        print("{} precision: {:.2f}%".format(self.ner_types[i],sub_p * 100))
        if self.ner_types[i] in no_sent_label:
            print('No sent with label')

    summary_dict = {}
    summary_dict["Mention F1"] = m_f1
    summary_dict["Mention recall"] = m_r
    summary_dict["Mention precision"] = m_p
    

    return util.make_summary(summary_dict), m_f1, m_p, m_r, pred_ners_list, predicted_sentences, self.ner_maps, golden_ners, predicted_sentences_ids, loss, total_gold_labels #/num
    
    
  def predict(self, session, is_training=False, evaluate_loss=False):
    def load_line(batch):
      return self.tensorize_example(batch, is_training, evaluate_loss), batch
    
    predict_file = self.config["predict_path"]
    with open(predict_file) as f:
      eval_examples = [json.loads(jsonline) for jsonline in f.readlines()]
      batch_size = self.config["batch_size"]
      batches = [eval_examples[x:x+batch_size] for x in range(0, len(eval_examples), batch_size)]
      self.eval_data = [load_line(batch) for batch in batches]

    print("Loaded {} examples to predict.".format(len(self.eval_data)))

    start_time = time.time()
    num_words = 0
    is_flat_ner = 'flat_ner' in self.config and self.config['flat_ner']
    pred_ners_list = []
    predicted_sentences = []
    predicted_sentences_ids = []

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      candidate_ner_scores = session.run(self.predictions, feed_dict=feed_dict)

      sentences = []
      ids = []
      #ners = []
      for s in example:
        num_words += sum(len(tok) for tok in s["sentence"])
        sentences.append(s["sentence"])
        ids.append(s["id"])
        #ners.append(s["ners"])
        
      pred_ners = self.get_pred_ner(sentences, candidate_ner_scores,is_flat_ner)
      pred_ners_list.append(pred_ners)
      predicted_sentences.append(sentences)
      predicted_sentences_ids.append(ids)
 
      
      #if example_num % 10 == 0:
      print("Predicted {}/{} examples.".format(example_num + 1, len(self.eval_data)))

    used_time = time.time() - start_time
    print("Time used: %d second, %.2f w/s " % (used_time, num_words*1.0/used_time))

    print('###########')
    sentence_labels = {}
    if "pt_law_10_NORMS" in self.config["log_dir"]:  #file to write predicted labels of NORMS model to be used in the 2 phased system for the SR and NE Model
        if os.path.exists("Norms_pred_test_ners.jsonl"):
            os.remove("Norms_pred_test_ners.jsonl")
        file = open("Norms_pred_test_ners.jsonl","w")
        
    #reading predicted labels and adding them on the sentence to later write all annotated sentences into table
    pred_final_results = {}
    sentence_batch = 0
    ners_for_2phased = {} #dictionary to save the predicted ners that will be used for the Hybrid and Two Phased model
    for sentences_ners in pred_ners_list:
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
            
            label = str(list(self.ner_maps.keys())[list(self.ner_maps.values()).index(sentence_ner[3])])
            sentence_labels[sent_id]['ners'].append([sentence_ner[1],sentence_ner[2],label])
            pred_final_results[sent_id][sentence_ner[1]] += '[' + label + sent[sentence_ner[1]] + ']'
            pred_final_results[sent_id][sentence_ner[2]] += '[' + label + sent[sentence_ner[2]] + ']'
            
            if "pt_law_10_NORMS" in self.config["log_dir"]: #saving in dict predicted norms when evaluating Norms model
                if sent_id not in ners_for_2phased.keys():
                    ners_for_2phased[sent_id] = []
                ners_for_2phased[sent_id].append([sentence_ner[1], sentence_ner[2], label])
        sentence_batch += 1
    
    if "pt_law_10_NORMS" in self.config["log_dir"]: #saving in file all predicted norms when evaluating Norms model
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
    if "withNorms" in self.config["log_dir"]: #adjust start and end of predicted spans considering tokens I_NORMLABEL and E_NORMLABEL were added into segment
        for id in sentence_labels.keys(): #for each sentence
            pos = 0
            count_norms_found = 0
            for token in sentence_labels[sent_id]["sentence"]: # for each token
                index = pos - count_norms_found
                if 'OBRIG' in token or  'RIGHT' in token or 'DEF' in token or 'INTRO' in token or 'LEFFECT' in token: #save index of added tokens
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

    test_json.close()

    return 'done'

