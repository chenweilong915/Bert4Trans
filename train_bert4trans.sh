python ./bert-master/create_pretraining_data.py --input_file=./geo_5_text.txt  --output_file=./tmp2/tf_examples.tfrecord --vocab_file=./vocab.txt --do_lower_case=False --max_seq_length=32 --max_predictions_per_seq=5 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5

python ./bert-master/run_pretraining.py \
  --input_file=./tmp2/tf_examples.tfrecord \
  --output_dir=./tmp2/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=./tmp2/pretraining_output/bert_config.json \
  --init_checkpoint=./tmp2/pretraining_output/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=32 \
  --max_predictions_per_seq=5 \
  --num_train_steps=5000 \
  --num_warmup_steps=20 \
  --learning_rate=2e-5
