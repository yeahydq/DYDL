# coding=utf-8
import random
import string
import os
import argparse
import xmlUtil
import sys
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# 生成几位数的验证码
number = 1
# 生成验证码图片的高度和宽度
size = (15, 25)
# 背景颜色，默认为白色
bgcolor = (255, 255, 255)
# 字体颜色，默认为蓝色
fontcolor = (0, 0, 255)
# 干扰线颜色。默认为红色
linecolor = (255, 0, 0)
# 是否要加入干扰线
draw_line = True
# 加入干扰线条数的上下限
# line_number = (0, 0)
line_number = (0, 2)
PARM_FONTS=['Arial.ttf',
       # 'Lantinghei.ttc',
       # 'platech.ttf',
       # 'platechar.ttf'
       ]

# 用来随机生成一个字符串
def gene_text():
    source = list(string.ascii_letters)
    source = list([str(i) for i in range(10)])
    for index in range(0, 10):
        source.append(str(index))
    return ''.join(random.sample(source, number))  # number是生成验证码的位数


# 用来绘制干扰线
def gene_line(draw, width, height):
    begin = (random.randint(0, width), random.randint(0, height))
    end = (random.randint(0, width), random.randint(0, height))
    draw.line([begin, end], fill=linecolor)


# 生成验证码
def gene_code(picID=0):
    width, height = size  # 宽和高
    image = Image.new('RGB', (width, height), bgcolor)  # 创建图片
    for fontName in PARM_FONTS:
        font = ImageFont.truetype(os.path.join(font_path,fontName), 25)  # 验证码的字体
        draw = ImageDraw.Draw(image)  # 创建画笔
        text = gene_text()  # 生成字符串
        font_width, font_height = font.getsize(text)
        draw.text(((width - font_width) / number, (height - font_height) / number), text,
                  font=font, fill=fontcolor)  # 填充字符串
        if draw_line:
            gene_line(draw, width, height)
        #image = image.transform((width+30,height+10), Image.AFFINE, (1,-0.3,0,-0.1,1,0),Image.BILINEAR)  #创建扭曲
        #image = image.transform((width + 20, height + 10), Image.AFFINE, (1, -0.3, 0, -0.1, 1, 0), Image.BILINEAR)  # 创建扭曲
        #image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 滤镜，边界加强
        jpgName="{}_{}_{}.jpg".format(fontName.replace('.ttf','').replace('.ttc',''),str(picID),str(text))

        image.save(os.path.join(JPEGPath, jpgName))
        write_Xml(JPEGPath,AnnotationsPath, jpgName, text)

def write_Xml(JPEGPath,AnnotationsPath, fileName,objectName):
    # Open original file
    tree = xmlUtil.read_xml("./annotation_template.xml")
    xmlUtil.change_node_text(xmlUtil.find_nodes(tree, "filename"), fileName)
    xmlUtil.change_node_text(xmlUtil.find_nodes(tree, "path"), os.path.join(JPEGPath,fileName))
    xmlUtil.change_node_text(xmlUtil.find_nodes(tree, "object/name"), str(objectName))

    xmlUtil.write_xml(tree, os.path.join(AnnotationsPath, "{}.xml".format(fileName.replace('.jpg',''))))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--font_path',
        help = 'path of the system front',
        # default = '/Library/Fonts/Arial.ttf'
        default= './Fonts/'
    )
    parser.add_argument(
        '--output',
        help = 'path of the output',
        default = '/tmp/captchars'
    )
    parser.add_argument(
        '--records',
        help = 'number of records to be generate',
        default = 100,
        type=int
    )

    args = parser.parse_args()
    arguments = args.__dict__

    # 字体的位置，不同版本的系统会有不同
    # font_path = '/Library/Fonts/Arial.ttf'
    #font_path = '/usr/share/fonts/dejavu/DejaVuSerif.ttf'
    font_path = arguments['font_path']


    CAPTCHA_IMAGE_FOLDER = arguments['output']
    # if the output directory does not exist, create it
    if not os.path.exists(CAPTCHA_IMAGE_FOLDER):
        os.makedirs(CAPTCHA_IMAGE_FOLDER)
    JPEGPath=os.path.join(CAPTCHA_IMAGE_FOLDER,'JPEGImages')
    AnnotationsPath=os.path.join(CAPTCHA_IMAGE_FOLDER,'Annotations')
    if not os.path.exists(JPEGPath):
        os.makedirs(JPEGPath)
    if not os.path.exists(AnnotationsPath):
        os.makedirs(AnnotationsPath)

    targetCount=int(arguments['records'])
    for x in range(int(arguments['records'])):
        gene_code(x)

