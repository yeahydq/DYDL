
pip install tensorflow-serving-api

export PYTHONPATH=${PYTHONPATH}:${PWD}/trainer/

rm -rf mnist_tfrecord/

# generate tfrecord
python ./genData.py

# check tfrecord
python ./readTfRecord.py


# CNN

export PYTHONPATH=${PYTHONPATH}:${PWD}/trainer/

rm -rf taxi_trained/
python -m trainer.task \
--train_data_paths=./sample/california_housing_train.csv \
  --eval_data_paths=./sample/eval.csv  \
  --output_dir=./taxi_trained \
  --train_steps=2000 \
  --eval_throttle_secs=60 \
  --job-dir=/tmp

tensorboard --logdir ./taxi_trained


# Predict
docker pull tensorflow/serving

docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source=/Users/dickye/Documents/codes/pycharm/solving_captchas_code_examples/tutorials/tfrecord_custom_estimator/taxi_trained/export/exporter,target=/models/tfrecord_custom_estimator -e MODEL_NAME=tfrecord_custom_estimator -t tensorflow/serving


python ./grpc_client.py \
  --image ./decode/8_Label_8.jpg \
  --model tfrecord_custom_estimator \
  --host localhost


  --signature_name classes

docker ps
docker kill


