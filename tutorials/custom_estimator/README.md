


export PYTHONPATH=${PYTHONPATH}:${PWD}/trainer/

rm -rf taxi_trained/
python -m trainer.task \
--train_data_paths=./sample/california_housing_train.csv \
  --eval_data_paths=./sample/eval.csv  \
  --output_dir=./taxi_trained \
  --train_steps=10000 \
  --job-dir=/tmp

tensorboard --logdir ./taxi_trained
