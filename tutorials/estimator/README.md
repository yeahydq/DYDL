


export PYTHONPATH=${PYTHONPATH}:${PWD}/trainer/

%%bash
python -m trainer.task \
--train_data_paths=./sample/california_housing_train.csv \
  --eval_data_paths=./sample/california_housing_train.csv  \
  --output_dir=./taxi_trained \
  --train_steps=1000 \
  --job-dir=/tmp

tensorboard --logdir ./taxi_trained