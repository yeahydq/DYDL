# -*- coding: UTF-8 -*-


import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions

# 新建模型，此处实际上是导入预训练模型
model = InceptionV3()
model.summary()



# 按照 InceptionV3 模型的默认输入尺寸，载入 demo1 图像
img = image.load_img('demo1.jpg', target_size=(299, 299))

# 提取特征
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 预测并输出概率最高的三个类别
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
