# INSTALL
1. pip install -r requirements.txt

# CONFIGURATION
1. config.yaml
1. model.yaml

# RUN
1. python trains.py

# INTRODUCTION
https://www.jianshu.com/p/b1a5427db6e2


# generate data
python genCaptchars.py --font_path /usr/share/fonts/dejavu/DejaVuSerif.ttf --output /tmp/generated_captcha_images_TRAIN --records 10000


python genCaptchars.py --font_path /usr/share/fonts/dejavu/DejaVuSerif.ttf --output /tmp/ec2-user/generated_captcha_images_TRAIN --records 100
