#!/bin/bash

python run_ft.py \
    --output_dir output/bert-base-uncased \
    --bert_pretrain_path pretrained/MWP-BERT_en \
    --data_dir ret_aug_data \
    --train_file mathqa_train.json \
    --finetune_from_trainset mathqa_train.json \
    --dev_file mathqa_val.json \
    --test_file mathqa_test.json \
    --schedule linear \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --n_epochs 100 \
    --warmup_steps 4000 \
    --n_save_ckpt 3 \
    --n_val 5 \
    --logging_steps 100 \
    --embedding_size 128 \
    --hidden_size 768 \
    --beam_size 5 \
    --dropout 0.5 \
    --seed 17
