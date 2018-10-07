git clone https://github.com/yeahydq/DYDL.git

source activate tensorflow_p36
rm -rf /tmp/tensorflow/mnist/logs/mnist_with_summaries*

cd DYDL/tutorials/image/mnist
nohup python mnist_with_summary.py --max_steps 100000 --dropout 0.8 &

nohup tensorboard --logdir=/tmp/tensorflow/mnist/logs/mnist_with_summaries &

# model check point:
/tmp/mnist_model
