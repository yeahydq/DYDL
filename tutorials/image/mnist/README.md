rm -rf /tmp/tensorflow/mnist/logs/mnist_with_summaries*

cd DYDL/tutorials/image/mnist
nohup python mnist_with_summary.py --max_steps 100000 --dropout 0.8 &

tensorboard --logdir=/tmp/tensorflow/mnist/logs/mnist_with_summaries

