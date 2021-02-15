export BERT_MODEL_PATH="./bert-large-portuguese-cased_tensorflow_checkpoint"
PYTHONPATH=. python extract_features.py --input_file="train.jsonl" --output_file=./bert_features.hdf5 --bert_config_file $BERT_MODEL_PATH/bert_config.json --init_checkpoint $BERT_MODEL_PATH/model.ckpt-1000000 --vocab_file  $BERT_MODEL_PATH/vocab.txt --do_lower_case=False --stride 1 --window-size 129
