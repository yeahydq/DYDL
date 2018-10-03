


export PYTHONPATH=${PYTHONPATH}:${PWD}/trainer/

%%bash
python -m trainer.task \
--train_data_paths=./sample/train.csv \
  --eval_data_paths=./sample/valid.csv  \
  --output_dir=./taxi_trained \
  --train_steps=1000 \
  --job-dir=/tmp

tensorboard --logdir ./taxi_trained