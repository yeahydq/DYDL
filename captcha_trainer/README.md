# INSTALL
1. pip install -r requirements.txt


# generate data
cd DYDL/captcha_trainer
python genCaptchars.py --font_path /usr/share/fonts/dejavu/DejaVuSerif.ttf --output /tmp/generated_captcha_images_TRAIN --records 1000


python genCaptchars.py --font_path /usr/share/fonts/dejavu/DejaVuSerif.ttf --output /tmp/generated_captcha_images_TEST --records 50


# CONFIGURATION
1. config.yaml
1. model.yaml

# RUN
1. source activate tensorflow_p36
2. cd DYDL/captcha_trainer
2. python trains.py
3. tensorboard --logdir=/model/capchar/model/
4. http://52.87.225.202:6006/

# INTRODUCTION
https://www.jianshu.com/p/b1a5427db6e2

(tensorflow_p36) [ec2-user@ip-172-31-23-184 captcha_trainer]$ python trains.py
Loading Configuration...
---------------------------------------------------------------------------------
CURRENT_CHECKPOINT: None
PROJECT_PATH /home/ec2-user/DYDL/captcha_trainer
MODEL_PATH: /home/ec2-user/DYDL/captcha_trainer/model/CaptchaName.model
COMPILE_MODEL_PATH: /home/ec2-user/DYDL/captcha_trainer/model/CaptchaName.pb
CHAR_SET_LEN: 36
IMAGE_WIDTH: 240, IMAGE_HEIGHT: 80, MAGNIFICATION: 2
IMAGE_ORIGINAL_COLOR: False
MAX_CAPTCHA_LEN 4
NEURAL NETWORK: CNNNet
3 LAYER CONV: [32, 64, 64], FULL_CONNECT: 1024
---------------------------------------------------------------------------------
WARNING:tensorflow:ParseError: 1:24 : Expected string but found: 'None'
WARNING:tensorflow:/home/ec2-user/DYDL/captcha_trainer/model/checkpoint: Checkpoint ignored
Can't load save_path when it is None.
10 Loss:  0.1557183
20 Loss:  0.13328937
30 Loss:  0.12984551
40 Loss:  0.12902172
50 Loss:  0.1270997
60 Loss:  0.12710486
70 Loss:  0.12725766
80 Loss:  0.12752604
90 Loss:  0.12690833
100 Loss:  0.12621957
EP Spend: 82.31s Train-Acc: 0.00%
110 Loss:  0.12662774
120 Loss:  0.1273409
130 Loss:  0.12714413
140 Loss:  0.12655067
150 Loss:  0.12673865

