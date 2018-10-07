# -*- coding:utf-8 -*-

__author__ = "David Chow"

# 将图片保存成 TFRecord  
import tensorflow as tf
import numpy as np

savedir = "./data/data.tfrecords"  # 希望在data/文件夹中生成“data.tfrecords"的TFRecord格式文件


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load(imgdir, width, height, method=0):
    # 准备一个 writer 用来写 TFRecord 文件  
    writer = tf.python_io.TFRecordWriter(savedir)
    imglist = open(imgdir, 'r')
    with tf.Session() as sess:
        for line in imglist:  # open 打开的文件返回对象是一个可迭代对象，直接用 for 迭代访问
            # 获得图片的路径和类型
            tmp = line.strip().split(' ')  # str.strip([chars])用于去除头尾的字符chars,为空时默认删除空白符；
            # str.split(' ')通过指定一个空格对字符串进行切片，返回分割后的字符串列表tmp
            imgpath = tmp[0]  # 字符串列表tmp中tmp[0]代表该图像的路径
            label = int(tmp[1])  # 字符串列表tmp中tmp[1]代表该图像的标签

            # 读取图片
            img = tf.gfile.FastGFile(imgpath, 'rb').read()
            # 解码图片（如果是 png 格式就使用tf.image.decode_png)）
            img = tf.image.decode_jpeg(img)

            # 图片归一化，[0,1],浮点类型数据。因为为了将图片数据能够保存到 TFRecord 结构体中，所以需要将其图片矩阵转换成 string，
            # 所以为了在使用时能够转换回来，这里确定下数据格式为 tf.float32
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)

            # 把图片转换成希望的大小，由于本例子中两张图片大小都是650*434，所以此步骤可以省略。要注意的时候resize_images中输入图片的宽、高顺序
            img = tf.image.resize_images(img, [height, width], method)

            # 执行op:image
            img = sess.run(img)

            # 将其图片矩阵转换成string
            img_raw = img.tostring()

            # 将数据整理成 TFRecord 需要的数据结构
            example = tf.train.Example(features=tf.train.Features(feature= \
                                                                      {'imge_raw': _bytes_feature(img_raw),
                                                                       'label': _int64_feature(label)}))

            # 写 TFRecord
            writer.write(example.SerializeToString())  # SerializeToString()作用:把example序列化为一个字符串,因为在写入到TFRcorde的时候,write方法的参数是字符串的.
    writer.close()


if __name__ == '__main__':
    load("./image_information.txt", 650, 434)
