# -*- coding: UTF-8 -*-
import glob
import os

latest_file = max(glob.glob('weights/conv_pool_cnn*'), key=os.path.getctime)
print(latest_file)